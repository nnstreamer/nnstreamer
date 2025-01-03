#ifndef __GST_TENSOR_FILTER_LLM_H__
#define __GST_TENSOR_FILTER_LLM_H__

#include <gst/gst.h>
#include <tensor_common.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_FILTER_LLM (gst_tensor_filter_llm_get_type ())
#define GST_TENSOR_FILTER_LLM(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_FILTER_LLM, GstTensorFilterLLM))
#define GST_TENSOR_FILTER_LLM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_FILTER_LLM, GstTensorFilterLLMClass))
#define GST_IS_TENSOR_FILTER_LLM(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_TENSOR_FILTER_LLM))
#define GST_IS_TENSOR_FILTER_LLM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_TENSOR_FILTER_LLM))
typedef struct _GstTensorFilterLLM GstTensorFilterLLM;
typedef struct _GstTensorFilterLLMClass GstTensorFilterLLMClass;

/**
 * @brief GstTensorFilterLLM data structure.
 */
struct _GstTensorFilterLLM
{
  GstElement element; /**< parent object */

  GstPad *sinkpad; /**< sink pad */
  GstPad *srcpad; /**< src pad */
  GstTensorsConfig in_config; /**< input tensors config */
};

/**
 * @brief GstTensorFilterLLMClass data structure.
 */
struct _GstTensorFilterLLMClass
{
  GstElementClass parent_class; /**< parent class */
};

/**
 * @brief Function to get type of tensor_filter_llm.
 */
GType gst_tensor_filter_llm_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_FILTER_LLM_H__ */
