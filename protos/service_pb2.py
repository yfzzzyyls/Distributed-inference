# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: service.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rservice.proto\x12\x06protos\"G\n\rVerifyRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x14\n\x0crequest_type\x18\x02 \x01(\t\x12\x0c\n\x04text\x18\x03 \x01(\t\"o\n\x14VerifyResultResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x11\n\tnext_text\x18\x02 \x01(\t\x12\x15\n\rpassed_tokens\x18\x03 \x01(\x05\x12\x19\n\x11generate_finished\x18\x04 \x01(\x08\"S\n\x0bInitRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x14\n\x0crequest_type\x18\x02 \x01(\t\x12\x11\n\x04text\x18\x03 \x01(\tH\x00\x88\x01\x01\x42\x07\n\x05_text\"W\n\x13InitRequestResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x11\n\tnext_text\x18\x02 \x01(\t\x12\x19\n\x11generate_finished\x18\x03 \x01(\x08\"9\n\rDeleteRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x14\n\x0crequest_type\x18\x02 \x01(\t\"+\n\x15\x44\x65leteRequestResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\t2\xda\x01\n\x0c\x42\x61tchService\x12\x44\n\rVerifyRequest\x12\x15.protos.VerifyRequest\x1a\x1c.protos.VerifyResultResponse\x12?\n\x0bInitRequest\x12\x13.protos.InitRequest\x1a\x1b.protos.InitRequestResponse\x12\x43\n\rDeleteRequest\x12\x13.protos.InitRequest\x1a\x1d.protos.DeleteRequestResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_VERIFYREQUEST']._serialized_start=25
  _globals['_VERIFYREQUEST']._serialized_end=96
  _globals['_VERIFYRESULTRESPONSE']._serialized_start=98
  _globals['_VERIFYRESULTRESPONSE']._serialized_end=209
  _globals['_INITREQUEST']._serialized_start=211
  _globals['_INITREQUEST']._serialized_end=294
  _globals['_INITREQUESTRESPONSE']._serialized_start=296
  _globals['_INITREQUESTRESPONSE']._serialized_end=383
  _globals['_DELETEREQUEST']._serialized_start=385
  _globals['_DELETEREQUEST']._serialized_end=442
  _globals['_DELETEREQUESTRESPONSE']._serialized_start=444
  _globals['_DELETEREQUESTRESPONSE']._serialized_end=487
  _globals['_BATCHSERVICE']._serialized_start=490
  _globals['_BATCHSERVICE']._serialized_end=708
# @@protoc_insertion_point(module_scope)
