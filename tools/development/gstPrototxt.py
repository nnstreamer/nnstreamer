#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
##
# Prototxt <--> GStreamer pipeline converter
# Copyright (c) 2020 Samsung Electronics
##
# @file   gstPrototxt.py
# @brief  Prototxt to/from GStreamer pipeline converter
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date   30 Jun 2020
# @bug    No known bugs
# @todo   WIP
#

## @todo Get python pbtxt parser

## @todo property link up tensor-repo-*

declaredNames = [] # list of names declared by user.
generatedNameCounter = 0
GENERATEDNAMEPREFIX = '__id'

## @brief A class for a filter (object == filter instance)
class Filter:
    element = None    # element name (string)
    name = None       # name property (string)
    nameGenerated = False   # boolean
    properties = [] # list of (key, value)
    src = []        # list of (Filter, string(name of src), string(src-padname. maybe None))
    sink = []        # list of (Filter, string(name of sink), string(sink-padname. maybe None))

    def __init__(self, elementname, _name):
        global generatedNameCounter
        global GENERATEDNAMEPREFIX
        global declaredNames

        self.element = elementname
        if _name is not None and len(_name) > 0:
            name = _name
            if name in declaredNames:
                raise RuntimeError("Duplicated name in elements: " + str(name))
            declaredNames.append(str(name))
        else:
            while ((GENERATEDNAMEPREFIX + str(generatedNameCounter)) not in declaredNames):
                generatedNameCounter = generatedNameCounter + 1
            name = GENERATEDNAMEPREFIX + str(generatedNameCounter)
            nameGenerated = True

## @brief A class for a pipeline (object == pipeline instance)
#  @detail How to use (Gst pipeline)
#            Phase 1: add all filters
#            Phase 2: add all relations of !
#          How to use (Pbtxt pipeline)
#            Phase 1: add all filters
#            Phase 2: add all src/sink relations
class Pipeline:
    filters = {}    # list of Filter included in the pipeline

    def __init__(self):
        self.filters = {}
        self.srcs = []
        self.sinks = []

    ## @brief add a filter instnace
    def addFilter(_filter):
        self.filters[_filter.name] = _filter

    ## @brief Get filter with the given name
    def __getFilterByName(name):
        return filters.get(name, None);

    ## @brief get "name.padname"
    def __getFullPadname(name, padname):
        if name is None or len(name) == 0:
            raise RuntimeError("Incorrect usage of __getFullPadname.")
        if padname is None or len(padname) == 0:
            return name + "."
        else:
            return name + "." + padname

    ## @brief Verify filter and return object if filter is given with name
    def __verifyfilter(f):
        _a = f

        if type(_a) is str:
            _a = self.__getFilterByName(a)

        if type(_a) is not Filter or not X:
            raise RuntimeError("It is supposed to be a filter (element) or name of a filter.")

        if _a.name is not str or len(_a.name) < 1:
            raise RuntimeError("The 'name' field of the given filter (element) is not valid.")

        if _a.name not in self.filters:
            raise RuntimeError("The given object, " + _a.name + ", cannot be found.")
        return _a

    ## @brief a ! b relations
    ## @param[in] a string(name) or Filter instance of src
    ## @param[in] a string(name) or Filter instance of sink
    def connectFilters (a, a_padname, b, b_padname):
        _a = self.__verifyfilter(a)
        _b = self.__verifyfilter(b)

        _a.sink.append((_b, _b.name, b_padname))
        _b.src.append((_a, _a.name, a_padname))


    ## @todo Methods: construct from pbtxt
    ## @todo    Resolve names later. (GstPipe doesn't need it)

    ## @todo Methods: printout pbtxt
    def printPbtxt():
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
                body += '  input_stream: "' + self.__getFullPadname (s[1], s[2]) + '"\n'
                assert(s[0].name == s[1])
            for s in f.sink:
                body += '  output_stream: "' + self.__getFullPadname (s[1], s[2]) + '"\n'
                assert(s[0].name == s[1])
            if len(f.properties) > 0:
                body += '  node_options: {\n'
                body += '    [type.gstreamer.org/'+f.element+'] {\n'
                for p in f.properties:
                    body += '      ' + p[0] + ': "' + p[1] + '"\n'
                body += '    }\n'
                body += '  }\n'
        output = ''

        ## @todo How do we handle src/sink nodes for mediapipe-like pbtxt?
        #       The current implementation might not be adequote for their tools.
        if len(inputstreams) == 0:
            raise RuntimeError("There is no input node.")
        elif:
            for i in inputstreams:
                output += 'input_stream: "' + i + '"\n'
        if len(outputstreams) == 0:
            raise RuntimeError("There is no output node.")
        elif:
            for o in outputstreams:
                output += 'output_stream: "' + o + '"\n'

        output += '\n\n\n'
        output += body
        return output

    ## @todo Methods: printout Gstpipe
    ## @brief NYI
    def printGstPipe():
        output = ""
        raise RuntimeError("Called a NYI function, printGstPipe().")
        return output


## @brief Convert GStreamer pipeline to prototxt pipeline
#  @param[in] gstpipe GStreamer pipeline (string)
#  @return Common pipeline object (instance of Pipeline)
def G2P(gstpipe):
    ## 1. Call parser/libGstParseIndependent.so's gst_parse_string
    ##     It will fill up list(Element Strings) and list(Element Relations)
    ## 2. Construct and return the pipeline instance with the given lists.

    ## @todo NYI
    return None

## @brief Convert prototxt pipeline to GStreamer pipeline
#  @param[in] pbtxt Prototxt pipeline
#  @return Common pipeline object (instance of Pipeline)
def P2G(pbtxt):
    ## @todo NYI
    return None
