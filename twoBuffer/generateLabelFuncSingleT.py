import onnx
from omlt.io import write_onnx_model_with_bounds,load_onnx_neural_network
from omlt import OmltBlock, OffsetScaling
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
from pyomo.environ import *
#import time
import os, time, random

pytorch_model='./twoBuffer_pinnc.onnx'
onnx_model = onnx.load(pytorch_model)
network_definition = load_onnx_neural_network(onnx_model)
for layer_id, layer in enumerate(network_definition.layers):
    print(f"{layer_id}\t{layer}\t{layer.activation}")
formulation = FullSpaceNNFormulation(network_definition)

##############################################################################

def calculateFunc(uncertainParameterList):
    theta1=uncertainParameterList[0]
    qin1=uncertainParameterList[1]
    if qin1-20*(theta1-0.3)**2-0.3<=0:
        model = ConcreteModel()

        model.theta1=theta1
        model.qin1=qin1

        model.h1_1in=4.0
        model.h2_1in=6.0
        model.t1 = 50.0

        model.nn1= OmltBlock()
        model.nn1.build_formulation(formulation)

        model.h1_2in=Var(initialize=5,within=Reals)
        model.h2_2in=Var(initialize=5,within=Reals)
        model.u1in=Var(initialize=0.2,within=Reals, bounds=(0, 0.4))

        #########################################################################
        @model.Constraint()
        def connect_input1_1(mdl):
            return mdl.h1_1in == mdl.nn1.inputs[0]

        @model.Constraint()
        def connect_input1_2(mdl):
            return mdl.h2_1in == mdl.nn1.inputs[1]

        @model.Constraint()
        def connect_input1_3(mdl):
            return mdl.u1in == mdl.nn1.inputs[2]

        @model.Constraint()
        def connect_input1_4(mdl):
            return mdl.theta1  == mdl.nn1.inputs[3]

        @model.Constraint()
        def connect_input1_5(mdl):
            return mdl.qin1  == mdl.nn1.inputs[4]

        @model.Constraint()
        def connect_input1_6(mdl):
            return mdl.t1 == mdl.nn1.inputs[5]

        @model.Constraint()
        def connect_output1_1(mdl):
            return mdl.h1_2in == mdl.nn1.outputs[0]

        @model.Constraint()
        def connect_output1_2(mdl):
            return mdl.h2_2in == mdl.nn1.outputs[1]
        ###############################################


        #########################################################################
        model.Result=Var(initialize=0,within=Reals)

        model.obj1 = Objective(expr=model.Result, sense=minimize)

        model.a2_1 = Constraint(expr = model.h1_2in-6-model.Result<=0)
        model.a2_2 = Constraint(expr = model.h2_2in-6-model.Result<=0)

        model.b2_1 = Constraint(expr = 4-model.h1_2in-model.Result<=0)
        model.b2_2 = Constraint(expr = 4-model.h2_2in-model.Result<=0)
        
        timeStart=time.time()
        opt=SolverFactory('ipopt', executable='./ipopt')
        results = opt.solve(model)
        timeEnd=time.time()
        slackVarValue=value(model.obj1)
        print(uncertainParameterList,'Profit = ',slackVarValue,'time consumption = ',timeEnd-timeStart)
        if slackVarValue<=0:
            flag=1
        else:
            flag=0
        return ([theta1,qin1,flag])
        #print('time consumption = ',timeEnd-timeStart)
    else:
        return ([theta1,qin1,0])


def labelFunc(uncertainParameterList):
    print('Run task (%s)...' % (os.getpid()))
    singleResult=calculateFunc(uncertainParameterList)
    print('Finish task (%s)...' % (os.getpid()))
    return singleResult



# def long_time_task(name):
#     print('Run task %s (%s)...' % (name, os.getpid()))
#     print(pytorch_model)
#     start = time.time()
#     time.sleep(random.random() * 3)
#     end = time.time()
#     print('Task %s runs %0.2f seconds.' % (name, (end - start)))

#calculateFunc(0.5)