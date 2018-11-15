tensor_aggregator_sources = [
    'tensor_aggregator.c'
]

tensor_aggregatorOBJ = static_library('tensor_aggregator',
    tensor_aggregator_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_aggregatorOBJ

