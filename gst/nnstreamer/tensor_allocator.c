/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Junhwan Kim <jejudo.kim@samsung.com>
 *
 * @file    tensor_allocator.c
 * @date    12 May 2021
 * @brief   Allocator for memory alignment
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */

#include <gst/gst.h>
#include "nnstreamer_plugin_api.h"

#define GST_TENSOR_ALLOCATOR "GstTensorAllocator"

static gsize gst_tensor_allocator_alignment = 0;

/**
 * @brief struct for type GstTensorAllocator
 */
typedef struct
{
  GstAllocator parent;
} GstTensorAllocator;

/**
 * @brief struct for class GstTensorAllocatorClass
 */
typedef struct
{
  GstAllocatorClass parent_class;
} GstTensorAllocatorClass;


static GType gst_tensor_allocator_get_type (void);
G_DEFINE_TYPE (GstTensorAllocator, gst_tensor_allocator, GST_TYPE_ALLOCATOR);

/**
 * @brief   allocation wrapper that binds alignment parameter
 */
static GstMemory *
_alloc (GstAllocator * allocator, gsize size, GstAllocationParams * params)
{
  GstAllocator *sysmem_alloc;
  GstAllocatorClass *sysmem_aclass;
  GstAllocationParams *_params;
  GstMemory *mem;

  sysmem_alloc = gst_allocator_find (GST_ALLOCATOR_SYSMEM);
  sysmem_aclass = GST_ALLOCATOR_GET_CLASS (sysmem_alloc);
  _params = gst_allocation_params_copy (params);
  _params->align = gst_tensor_allocator_alignment;

  mem = sysmem_aclass->alloc (allocator, size, _params);

  gst_allocation_params_free (_params);
  gst_object_unref (sysmem_alloc);
  return mem;
}

/**
 * @brief class initization for GstTensorAllocatorClass
 */
static void
gst_tensor_allocator_class_init (GstTensorAllocatorClass * klass)
{
  GstAllocatorClass *allocator_class, *sysmem_aclass;
  GstAllocator *sysmem_alloc;

  allocator_class = (GstAllocatorClass *) klass;
  sysmem_alloc = gst_allocator_find (GST_ALLOCATOR_SYSMEM);
  sysmem_aclass = GST_ALLOCATOR_GET_CLASS (sysmem_alloc);

  allocator_class->alloc = _alloc;
  allocator_class->free = sysmem_aclass->free;

  gst_object_unref (sysmem_alloc);
}

/**
 * @brief initialzation for GstTensorAllocator
 */
static void
gst_tensor_allocator_init (GstTensorAllocator * allocator)
{
  GstAllocator *sysmem_alloc, *alloc;

  alloc = GST_ALLOCATOR_CAST (allocator);
  sysmem_alloc = gst_allocator_find (GST_ALLOCATOR_SYSMEM);

  alloc->mem_type = sysmem_alloc->mem_type;
  alloc->mem_map = sysmem_alloc->mem_map;
  alloc->mem_unmap = sysmem_alloc->mem_unmap;
  alloc->mem_copy = sysmem_alloc->mem_copy;
  alloc->mem_share = sysmem_alloc->mem_share;
  alloc->mem_is_span = sysmem_alloc->mem_is_span;

  gst_object_unref (sysmem_alloc);
}

/**
 * @brief set alignment that default allocator would align to
 * @param alignment bytes of alignment
 */
void
gst_tensor_alloc_init (gsize alignment)
{
  GstAllocator *allocator;

  gst_tensor_allocator_alignment = alignment;

  /* no alignment */
  if (alignment == 0) {
    gst_allocator_set_default (gst_allocator_find (GST_ALLOCATOR_SYSMEM));
    return;
  }

  allocator = gst_allocator_find (GST_TENSOR_ALLOCATOR);
  /* allocator already set */
  if (allocator == NULL) {
    allocator = g_object_new (gst_tensor_allocator_get_type (), NULL);
    gst_allocator_register (GST_TENSOR_ALLOCATOR, gst_object_ref (allocator));
  }
  gst_allocator_set_default (allocator);
}
