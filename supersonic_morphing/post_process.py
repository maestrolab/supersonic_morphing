# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:36:46 2014

@author: Pedro
"""

from abaqus import *
from abaqusConstants import *
import visualization
from viewerModules import *

import numpy as np

def findOutputSet(ModelName,StepName,InstanceName,SetName, Output='COORD'):
    output=[]
    time = []
    odbName = ModelName+'.odb'

    odb = visualization.openOdb(odbName)
    for i in range(len(odb.steps[StepName].frames)):
        output_frame = []
        lastFrame = odb.steps[StepName].frames[i]
        print(i, lastFrame.frameValue)
        time.append(lastFrame.frameValue)
        coordset = odb.rootAssembly.instances[InstanceName.upper()].nodeSets[SetName.upper()]

        # Retrieve Y-displacements at the splines/connectors
        dispField = lastFrame.fieldOutputs[Output]

        dFieldpTip = dispField.getSubset(region=coordset)

        for i in range(len(dFieldpTip.values)):
            output_frame.append(dFieldpTip.values[i].data)
        output.append(output_frame)
    odb.close()

    return np.array(output), np.array(time)

if __name__ == '__main__':
    import pickle as p
    ModelName = 'small_simple_1'
    Steps = ['Step-1', 'Step-2', 'Step-3', 'Step-4', 'Step-5']#, 'Step-6', 'Step-7']
    InstanceName = 'Part-3-1'
    SetName1 = 'Whole-surf'
    SetName2 = 'Middle-point'
    outputNames = ['NT11', 'COORD', 'E', 'U', 'SDV2']
    coordinates = {}
    temperatures = {}
    output = {'Time':{}}
    mid_output = {'Time':{}}
    for outputName in outputNames:
        output[outputName] = {}
        mid_output[outputName] = {}
        for StepName in Steps:
            output_i, time = findOutputSet(ModelName,StepName,InstanceName,
                                         SetName1, outputName)
            output_i2, time2 = findOutputSet(ModelName,StepName,InstanceName,
                                         SetName2, outputName)
            output[outputName][StepName] = output_i
            mid_output[outputName][StepName] = output_i2
            output['Time'][StepName] = time
            mid_output['Time'][StepName] = time2

    f = open('outputs_small_simple_test.p', 'wb')
    p.dump(output, f)
    f.close()
    f = open('mid_outputs_small_simple_test.p', 'wb')
    p.dump(mid_output, f)
    f.close()
