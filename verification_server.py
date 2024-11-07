import grpc
from concurrent import futures
import time
import redis
import json
import random
import protos.service_pb2 as service_pb2
import protos.service_pb2_grpc as service_pb2_grpc

class SchedulingStrategy:
    """调度策略接口."""
    def select_queue(self, available_queues, redis_client):
        """选择合适的队列."""
        raise NotImplementedError


class RandomSchedulingStrategy(SchedulingStrategy):
    """随机选择队列的调度策略."""
    def __init__(self):
        self.queue_metrics = {}  # 用于存储每个队列的处理时间和正确率
        
    def update_metrics(self, queue_name, processing_time, correct_rate=0):
        """更新队列的处理时间和正确率."""
        if queue_name not in self.queue_metrics:
            self.queue_metrics[queue_name] = {
                'total_time': 0,
                'total_correct_rate': 0,
                'task_count': 0,
            }
        metrics = self.queue_metrics[queue_name]
        metrics['total_time'] += processing_time
        metrics['total_correct_rate'] += correct_rate
        metrics['task_count'] += 1

    def select_queue(self, available_queues, redis_client):
        return random.choice(available_queues)


class LoadBalancingSchedulingStrategy(SchedulingStrategy):
    """选择负载最小的队列的调度策略."""
    def select_queue(self, available_queues, redis_client):
        min_queue = None
        min_length = float('inf')

        for queue in available_queues:
            queue_length = redis_client.llen(queue)  # 获取队列长度
            if queue_length < min_length:
                min_length = queue_length
                min_queue = queue

        return min_queue if min_queue is not None else random.choice(available_queues)


class LoadBalancingWithMetricsSchedulingStrategy(SchedulingStrategy):
    """根据负载、处理时间和正确率选择队列的调度策略."""

    def __init__(self):
        self.queue_metrics = {}  # 用于存储每个队列的处理时间和正确率

    def update_metrics(self, queue_name, processing_time, correct_rate):
        """更新队列的处理时间和正确率."""
        if queue_name not in self.queue_metrics:
            self.queue_metrics[queue_name] = {
                'total_time': 0,
                'total_correct_rate': 0,
                'task_count': 0,
            }
        metrics = self.queue_metrics[queue_name]
        metrics['total_time'] += processing_time
        metrics['total_correct_rate'] += correct_rate
        metrics['task_count'] += 1

    def select_queue(self, available_queues, redis_client):
        best_queue = None
        best_score = float('-inf')

        for queue in available_queues:
            queue_length = redis_client.llen(queue)  # 获取队列长度
            avg_time = self.queue_metrics[queue]['total_time'] / self.queue_metrics[queue]['task_count'] if self.queue_metrics[queue]['task_count'] > 0 else 0
            avg_correct_rate = self.queue_metrics[queue]['total_correct_rate'] / self.queue_metrics[queue]['task_count'] if self.queue_metrics[queue]['task_count'] > 0 else 0

            # 计算评分：负载越小，处理时间越短，正确率越高，得分越高
            score = (1 / (queue_length + 1)) + (1 / (avg_time + 1)) + avg_correct_rate  # 添加1以避免除零错误

            if score > best_score:
                best_score = score
                best_queue = queue

        return best_queue if best_queue is not None else random.choice(available_queues)


class TaskManager:
    """负责选择合适的 Verification Server 队列。"""
    
    def __init__(self, available_queues, strategy, redis_host='localhost', redis_port=6379):
        self.available_queues = available_queues  # 可用的 Verification Server 队列列表
        self.strategy = strategy  # 调度策略实例
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)  # 连接到 Redis

    def select_queue(self):
        """选择合适的 Verification Server 队列."""
        return self.strategy.select_queue(self.available_queues, self.redis)


class VerificationService(service_pb2_grpc.BatchServiceServicer):
    """gRPC 服务类，监听客户端请求并处理任务。"""

    def __init__(self, task_manager):
        self.task_manager = task_manager
        self.redis = redis.Redis(host='localhost', port=6379, db=0)  # 连接 Redis
        self.queue_of_task = {}

    def VerifyRequest(self, request, context):
        # 使用 TaskManager 选择合适的队列
        if request.request_id not in self.queue_of_task:
            self.queue_of_task[request.request_id] = self.task_manager.select_queue()
        queue_name = self.queue_of_task.get(request.request_id)

        # 将任务数据序列化为 JSON 并推送到 Redis 队列
        task_data = {
            "request_id": request.request_id,
            "request_type": request.request_type,
            "text": request.text
        }
        self.redis.rpush(queue_name, json.dumps(task_data))
        print(f"VerifyRequest: {task_data}")
        # 等待处理后的结果
        result_queue_name = f"result_queue_{request.request_id}"
        start_time = time.time()  # 开始时间
        try:
            # 从 Redis 获取结果，使用 BLPOP 阻塞等待模式，超时为 30 秒
            result = self.redis.blpop(result_queue_name, timeout=30)
            if result:
                # 解析处理完成的结果
                verified_data = json.loads(result[1])
                status = verified_data.get("status", "success")
                if status == "error":
                    raise ValueError(verified_data.get("message", verified_data.message))
                next_text = verified_data.get("next_text", "")
                passed_tokens = verified_data.get("passed_tokens", 0)  # 获取通过的 token 数量
                correct_rate = verified_data.get("correct_rate", 0)  # 获取正确率
                generate_finished = verified_data.get("generate_finished", False)  # 获取通过的 token 数量
                processing_time = time.time() - start_time  # 计算处理时间
                self.task_manager.strategy.update_metrics(queue_name, processing_time, correct_rate)
                # 记录处理时间和验证正确率
                print(f"Processed task {request.request_id} with time: {processing_time:.2f}s and correct rate: {correct_rate:.2f}")
            else:
                raise TimeoutError("Task processing timed out.")
        except Exception as e:
            context.set_details(f"Error processing request: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.VerifyResultResponse(request_id=request.request_id, next_text="")
        
        return service_pb2.VerifyResultResponse(request_id=request.request_id, passed_tokens=passed_tokens, next_text=next_text, generate_finished=generate_finished)

    def InitRequest(self, request, context):
        # 使用 TaskManager 选择合适的队列
        if request.request_id not in self.queue_of_task:
            self.queue_of_task[request.request_id] = self.task_manager.select_queue()
        queue_name = self.queue_of_task.get(request.request_id)
        # 将任务数据序列化为 JSON 并推送到 Redis 队列
        task_data = {
            "request_id": request.request_id,
            "request_type": request.request_type,
            "text": request.text if request.text != "" else None
        }
        print(f"InitRequest: {task_data}")
        self.redis.rpush(queue_name, json.dumps(task_data))
        
        # 等待处理后的结果
        result_queue_name = f"result_queue_{request.request_id}"
        start_time = time.time()  # 开始时间
        try:
            # 从 Redis 获取结果，使用 BLPOP 阻塞等待模式，超时为 30 秒
            result = self.redis.blpop(result_queue_name, timeout=30)
            if result:
                # 解析处理完成的结果
                inited_data = json.loads(result[1])
                status = inited_data.get("status", "success")
                if status == "error":
                    raise ValueError(inited_data.get("message", inited_data.message))
                next_text = inited_data.get("next_text", "")
                generate_finished = inited_data.get("generate_finished", False)  # 获取通过的 token 数量
                processing_time = time.time() - start_time  # 计算处理时间
                self.task_manager.strategy.update_metrics(queue_name, processing_time, correct_rate=0)
                # 记录处理时间和验证正确率
                print(f"Processed task {request.request_id} with time: {processing_time:.2f}s.")
            else:
                raise TimeoutError("Task processing timed out.")
        except Exception as e:
            context.set_details(f"Error processing request: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.InitRequestResponse(request_id=request.request_id, verified_text="")
        
        return service_pb2.InitRequestResponse(request_id=request.request_id, next_text=next_text, generate_finished=generate_finished)

    def DeleteRequest(self, request, context):
        # 使用 TaskManager 选择合适的队列
        if request.request_id not in self.queue_of_task:
            self.queue_of_task[request.request_id] = self.task_manager.select_queue()
        queue_name = self.queue_of_task.get(request.request_id)

        # 将任务数据序列化为 JSON 并推送到 Redis 队列
        task_data = {
            "request_id": request.request_id,
            "request_type": request.request_type,
        }
        print(f"DeleteRequest: {task_data}")
        self.redis.rpush(queue_name, json.dumps(task_data))
        
        # 等待处理后的结果
        result_queue_name = f"result_queue_{request.request_id}"
        try:
            # 从 Redis 获取结果，使用 BLPOP 阻塞等待模式，超时为 30 秒
            result = self.redis.blpop(result_queue_name, timeout=30)
            if result:
                # 解析处理完成的结果
                deleted_data = json.loads(result[1])
                self.queue_of_task.pop(request.request_id, None)
                status = deleted_data.get("status", "success")
                if status == "error":
                    raise ValueError(deleted_data.get("message", deleted_data.message))
                # 记录处理时间和验证正确率
                print(f"Deleted task {request.request_id}.")
            else:
                raise TimeoutError("Task processing timed out.")
        except Exception as e:
            context.set_details(f"Error processing request: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.DeleteRequestResponse(request_id=request.request_id)
        
        return service_pb2.DeleteRequestResponse(request_id=request.request_id)
    

def serve():
    # 可用的 Verification Server 队列列表
    available_queues = ["task_queue_0", "task_queue_1"]
    
    # 选择调度策略
    strategy = RandomSchedulingStrategy()  # 或者使用 RandomSchedulingStrategy()
    
    task_manager = TaskManager(available_queues, strategy)
    
    # 启动 gRPC 服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    batch_service = VerificationService(task_manager)
    service_pb2_grpc.add_BatchServiceServicer_to_server(batch_service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051.")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
