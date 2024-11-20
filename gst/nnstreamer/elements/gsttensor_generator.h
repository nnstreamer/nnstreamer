#ifndef __GST_TENSOR_GENERATOR_H__
#define __GST_TENSOR_GENERATOR_H__

#include <gst/gst.h>
#include <tensor_common.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_GENERATOR (gst_tensor_generator_get_type ())
#define GST_TENSOR_GENERATOR(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_GENERATOR, GstTensorGenerator))
#define GST_TENSOR_GENERATOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_GENERATOR, GstTensorGeneratorClass))
#define GST_IS_TENSOR_GENERATOR(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_TENSOR_GENERATOR))
#define GST_IS_TENSOR_GENERATOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_TENSOR_GENERATOR))
typedef struct _GstTensorGenerator GstTensorGenerator;
typedef struct _GstTensorGeneratorClass GstTensorGeneratorClass;

/**
 * @brief GstTensorGenerator data structure.
 */
struct _GstTensorGenerator
{
  GstElement element; /**< parent object */

  GstPad *sinkpad; /**< sink pad */
  GstPad *srcpad; /**< src pad */
};

/**
 * @brief GstTensorGeneratorClass data structure.
 */
struct _GstTensorGeneratorClass
{
  GstElementClass parent_class; /**< parent class */
};

/**
 * @brief Function to get type of tensor_generator.
 */
GType gst_tensor_generator_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_GENERATOR_H__ */
