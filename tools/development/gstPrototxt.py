#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
##
# Prototxt <--> GStreamer pipeline converter
# Copyright (c) 2020 Samsung Electronics
#
# @file   gstPrototxt.py
# @brief  Prototxt to/from GStreamer pipeline converter
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date   30 Jun 2020
# @bug    No known bugs
# @todo   WIP
# @todo   Get python pbtxt parser
# @todo   property link up tensor-repo-*


##
# @brief A class for a filter (object == filter instance)
class Filter:
    element = None  # element name (string)
    name = None  # name property (string)
    nameGenerated = False  # boolean
    properties = []  # list of (key, value)
    src = []  # list of (Filter, string(name of src), string(src-padname. maybe None))
    sink = []  # list of (Filter, string(name of sink), string(sink-padname. maybe None))

    _generated = 0  # Internal, total number of generated names.
    _declared_names = []  # Internal, list of names declared by user.

    def __init__(self, _element, _name=None):
        if _name is not None and len(_name) > 0:
            if _name in Filter._declared_names:
                raise RuntimeError("Duplicated name in elements: " + str(_name))
            n = _name
        else:
            n = Filter.generate_element_name()

        self.element = str(_element)
        self.name = str(n)
        Filter._declared_names.append(str(n))

    @staticmethod
    def generate_element_name():
        def _gen_name(idx):
            return '__id' + str(idx)

        while _gen_name(Filter._generated) in Filter._declared_names:
            Filter._generated = Filter._generated + 1
        return _gen_name(Filter._generated)


##
# @brief A class for a pipeline (object == pipeline instance)
# @detail How to use (Gst pipeline)
#            Phase 1: add all filters
#            Phase 2: add all relations of !
#         How to use (Pbtxt pipeline)
#            Phase 1: add all filters
#            Phase 2: add all src/sink relations
class Pipeline:
    filters = {}  # list of Filter included in the pipeline

    def __init__(self):
        self.filters = {}
        self.srcs = []
        self.sinks = []

    ##
    # @brief add a filter instnace
    def addFilter(self, _filter):
        self.filters[_filter.name] = _filter

    ##
    # @brief Get filter with the given name
    def __getFilterByName(self, name):
        return self.filters.get(name, None)

    ##
    # @brief get "name.padname"
    def __getFullPadname(self, name, padname):
        if name is None or len(name) == 0:
            raise RuntimeError("Incorrect usage of __getFullPadname.")
        if padname is None or len(padname) == 0:
            return name + "."
        else:
            return name + "." + padname

    ##
    # @brief Verify filter and return object if filter is given with name
    def __verifyFilter(self, f):
        if type(f) is str:
            _a = self.__getFilterByName(f)
        else:
            _a = f

        if _a is None or type(_a) is not Filter:
            raise RuntimeError("It is supposed to be a filter (element) or name of a filter.")

        if _a.name is not str or len(_a.name) < 1:
            raise RuntimeError("The 'name' field of the given filter (element) is not valid.")

        if _a.name not in self.filters:
            raise RuntimeError("The given object, " + _a.name + ", cannot be found.")
        return _a

    ##
    # @brief a ! b relations
    # @param[in] a string(name) or Filter instance of src
    # @param[in] a string(name) or Filter instance of sink
    def connectFilters(self, a, a_padname, b, b_padname):
        _a = self.__verifyFilter(a)
        _b = self.__verifyFilter(b)

        _a.sink.append((_b, _b.name, b_padname))
        _b.src.append((_a, _a.name, a_padname))

    ##
    # @todo Methods: construct from pbtxt
    # @todo Resolve names later. (GstPipe doesn't need it)
    # @todo Methods: printout pbtxt
    def printPbtxt(self):
        body = ""
        inputstreams = []
        outputstreams = []

        for i in self.filters:
            f = self.filters[i]
            body += 'node {\n'
            body += '  calculator: "' + f.element + ":" + f.name + '"\n'

            if len(f.src) == 0:
                inputstreams.append(f.name)
            for s in f.src:
                body += '  input_stream: "' + self.__getFullPadname(s[1], s[2]) + '"\n'
                assert (s[0].name == s[1])
            for s in f.sink:
                body += '  output_stream: "' + self.__getFullPadname(s[1], s[2]) + '"\n'
                assert (s[0].name == s[1])
            if len(f.properties) > 0:
                body += '  node_options: {\n'
                body += '    [type.gstreamer.org/' + f.element + '] {\n'
                for p in f.properties:
                    body += '      ' + p[0] + ': "' + p[1] + '"\n'
                body += '    }\n'
                body += '  }\n'
        output = ''

        ##
        # @todo How do we handle src/sink nodes for mediapipe-like pbtxt?
        #       The current implementation might not be adequote for their tools.
        if len(inputstreams) == 0:
            raise RuntimeError("There is no input node.")
        else:
            for i in inputstreams:
                output += 'input_stream: "' + i + '"\n'
        if len(outputstreams) == 0:
            raise RuntimeError("There is no output node.")
        else:
            for o in outputstreams:
                output += 'output_stream: "' + o + '"\n'

        output += '\n\n\n'
        output += body
        return output

    ##
    # @todo Methods: printout Gstpipe
    # @brief NYI
    def printGstPipe(self):
        raise RuntimeError("Called a NYI function, printGstPipe().")


##
# @brief Convert GStreamer pipeline to prototxt pipeline
# @param[in] gstpipe GStreamer pipeline (string)
# @return Common pipeline object (instance of Pipeline)
def G2P(gstpipe):
    # @todo NYI
    # 1. Call parser/libGstParseIndependent.so's gst_parse_string
    #     It will fill up list(Element Strings) and list(Element Relations)
    # 2. Construct and return the pipeline instance with the given lists.
    return None


##
# @brief Convert prototxt pipeline to GStreamer pipeline
# @param[in] pbtxt Prototxt pipeline
# @return Common pipeline object (instance of Pipeline)
def P2G(pbtxt):
    # @todo NYI
    return None
