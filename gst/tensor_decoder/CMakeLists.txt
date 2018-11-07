set(DECODER_SOURCE tensordec.c tensordec_bounding_boxes_core.cc)
ADD_LIBRARY(tensor_decoderOBJ OBJECT ${DECODER_SOURCE})
