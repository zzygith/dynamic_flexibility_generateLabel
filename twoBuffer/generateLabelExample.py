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



# theta1=0.4
# qin1=0.3
# model = ConcreteModel()

# model.theta1=theta1
# model.qin1=qin1

# model.h1_1in=4.0
# model.h2_1in=6.0
# model.t1 = 50.0
# model.tM = 10.0

# model.nn1= OmltBlock()
# model.nn1.build_formulation(formulation)

# model.nn2= OmltBlock()
# model.nn2.build_formulation(formulation)

# model.h1_2in=Var(initialize=5,within=Reals)
# model.h2_2in=Var(initialize=5,within=Reals)

# model.h1_2inM=Var(initialize=5,within=Reals)
# model.h2_2inM=Var(initialize=5,within=Reals)

# model.u1in=Var(initialize=0.2,within=Reals, bounds=(0, 0.4))

# #########################################################################
# @model.Constraint()
# def connect_input1_1(mdl):
#     return mdl.h1_1in == mdl.nn1.inputs[0]

# @model.Constraint()
# def connect_input1_2(mdl):
#     return mdl.h2_1in == mdl.nn1.inputs[1]

# @model.Constraint()
# def connect_input1_3(mdl):
#     return mdl.u1in == mdl.nn1.inputs[2]

# @model.Constraint()
# def connect_input1_4(mdl):
#     return mdl.theta1  == mdl.nn1.inputs[3]

# @model.Constraint()
# def connect_input1_5(mdl):
#     return mdl.qin1  == mdl.nn1.inputs[4]

# @model.Constraint()
# def connect_input1_6(mdl):
#     return mdl.t1 == mdl.nn1.inputs[5]

# @model.Constraint()
# def connect_output1_1(mdl):
#     return mdl.h1_2in == mdl.nn1.outputs[0]

# @model.Constraint()
# def connect_output1_2(mdl):
#     return mdl.h2_2in == mdl.nn1.outputs[1]
# ###############################################

# # @model.Constraint()
# # def connect_input2_1(mdl):
# #     return mdl.h1_1in == mdl.nn1.inputs[0]

# # @model.Constraint()
# # def connect_input2_2(mdl):
# #     return mdl.h2_1in == mdl.nn1.inputs[1]

# # @model.Constraint()
# # def connect_input2_3(mdl):
# #     return mdl.u1in == mdl.nn1.inputs[2]

# # @model.Constraint()
# # def connect_input2_4(mdl):
# #     return mdl.theta1  == mdl.nn1.inputs[3]

# # @model.Constraint()
# # def connect_input2_5(mdl):
# #     return mdl.qin1  == mdl.nn1.inputs[4]

# # @model.Constraint()
# # def connect_input2_6(mdl):
# #     return mdl.tM == mdl.nn1.inputs[5]

# # @model.Constraint()
# # def connect_output2_1(mdl):
# #     return mdl.h1_2inM == mdl.nn1.outputs[0]

# # @model.Constraint()
# # def connect_output2_2(mdl):
# #     return mdl.h2_2inM == mdl.nn1.outputs[1]


# ####
# @model.Constraint()
# def connect_input2_1(mdl):
#     return mdl.h1_1in == mdl.nn2.inputs[0]

# @model.Constraint()
# def connect_input2_2(mdl):
#     return mdl.h2_1in == mdl.nn2.inputs[1]

# @model.Constraint()
# def connect_input2_3(mdl):
#     return mdl.u1in == mdl.nn2.inputs[2]

# @model.Constraint()
# def connect_input2_4(mdl):
#     return mdl.theta1  == mdl.nn2.inputs[3]

# @model.Constraint()
# def connect_input2_5(mdl):
#     return mdl.qin1  == mdl.nn2.inputs[4]

# @model.Constraint()
# def connect_input2_6(mdl):
#     return mdl.tM == mdl.nn2.inputs[5]

# @model.Constraint()
# def connect_output2_1(mdl):
#     return mdl.h1_2inM == mdl.nn2.outputs[0]

# @model.Constraint()
# def connect_output2_2(mdl):
#     return mdl.h2_2inM == mdl.nn2.outputs[1]

# ################################################
# model.Result=Var(initialize=0,within=Reals)

# model.obj1 = Objective(expr=model.Result, sense=minimize)

# model.a2_1 = Constraint(expr = model.h1_2in-6-model.Result<=0)
# model.a2_2 = Constraint(expr = model.h2_2in-6-model.Result<=0)

# model.b2_1 = Constraint(expr = 4-model.h1_2in-model.Result<=0)
# model.b2_2 = Constraint(expr = 4-model.h2_2in-model.Result<=0)

# model.a2_1M = Constraint(expr = model.h1_2inM-6-model.Result<=0)
# model.a2_2M = Constraint(expr = model.h2_2inM-6-model.Result<=0)

# model.b2_1M = Constraint(expr = 4-model.h1_2inM-model.Result<=0)
# model.b2_2M = Constraint(expr = 4-model.h2_2inM-model.Result<=0)



# timeStart=time.time()
# opt=SolverFactory('ipopt', executable='./ipopt')
# results = opt.solve(model)
# timeEnd=time.time()
# slackVarValue=value(model.obj1)
# print('Profit = ',slackVarValue,'time consumption = ',timeEnd-timeStart)

##################################################################################################################################################################################################################################################################################################################################################################################
uncertainParameterList=[0.4,0.3,0.4,0.2,0.3,0.4,0.3,0.4,0.3,0.4,0.3,0.4]

theta1=uncertainParameterList[0]
qin1=uncertainParameterList[1]

theta2=uncertainParameterList[2]
qin2=uncertainParameterList[3]

theta3=uncertainParameterList[4]
qin3=uncertainParameterList[5]

theta4=uncertainParameterList[6]
qin4=uncertainParameterList[7]

theta5=uncertainParameterList[8]
qin5=uncertainParameterList[9]

theta6=uncertainParameterList[10]
qin6=uncertainParameterList[11]


model = ConcreteModel()

model.theta1=theta1
model.qin1=qin1
model.theta2=theta2
model.qin2=qin2
model.theta3=theta3
model.qin3=qin3
model.theta4=theta4
model.qin4=qin4
model.theta5=theta5
model.qin5=qin5
model.theta6=theta6
model.qin6=qin6

######################################################
model.nn1= OmltBlock()
model.nn1.build_formulation(formulation)

model.nn2= OmltBlock()
model.nn2.build_formulation(formulation)

model.nn3= OmltBlock()
model.nn3.build_formulation(formulation)

model.nn4= OmltBlock()
model.nn4.build_formulation(formulation)

model.nn5= OmltBlock()
model.nn5.build_formulation(formulation)

model.nn6= OmltBlock()
model.nn6.build_formulation(formulation)

######################################################

model.t1 = 50.0
model.h1_1in=5.0
model.h2_1in=5.0
model.u1in=Var(initialize=0.2,within=Reals, bounds=(0, 0.4))


model.h1_1out=Var(initialize=5,within=Reals)
model.h2_1out=Var(initialize=5,within=Reals)
model.u2in=Var(initialize=0.2,within=Reals, bounds=(0, 0.4))

model.h1_2out=Var(initialize=5,within=Reals)
model.h2_2out=Var(initialize=5,within=Reals)
model.u3in=Var(initialize=0.2,within=Reals, bounds=(0, 0.4))       

model.h1_3out=Var(initialize=5,within=Reals)
model.h2_3out=Var(initialize=5,within=Reals)
model.u4in=Var(initialize=0.2,within=Reals, bounds=(0, 0.4))

model.h1_4out=Var(initialize=5,within=Reals)
model.h2_4out=Var(initialize=5,within=Reals)
model.u5in=Var(initialize=0.2,within=Reals, bounds=(0, 0.4))

model.h1_5out=Var(initialize=5,within=Reals)
model.h2_5out=Var(initialize=5,within=Reals)
model.u6in=Var(initialize=0.2,within=Reals, bounds=(0, 0.4))

model.h1_6out=Var(initialize=5,within=Reals)
model.h2_6out=Var(initialize=5,within=Reals)

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
    return mdl.h1_1out == mdl.nn1.outputs[0]

@model.Constraint()
def connect_output1_2(mdl):
    return mdl.h2_1out == mdl.nn1.outputs[1]
###############################################
@model.Constraint()
def connect_input2_1(mdl):
    return mdl.h1_1out == mdl.nn2.inputs[0]

@model.Constraint()
def connect_input2_2(mdl):
    return mdl.h2_1out == mdl.nn2.inputs[1]

@model.Constraint()
def connect_input2_3(mdl):
    return mdl.u2in == mdl.nn2.inputs[2]

@model.Constraint()
def connect_input2_4(mdl):
    return mdl.theta2  == mdl.nn2.inputs[3]

@model.Constraint()
def connect_input2_5(mdl):
    return mdl.qin2  == mdl.nn2.inputs[4]

@model.Constraint()
def connect_input2_6(mdl):
    return mdl.t1 == mdl.nn2.inputs[5]

@model.Constraint()
def connect_output2_1(mdl):
    return mdl.h1_2out == mdl.nn2.outputs[0]

@model.Constraint()
def connect_output2_2(mdl):
    return mdl.h2_2out == mdl.nn2.outputs[1]
###############################################
@model.Constraint()
def connect_input3_1(mdl):
    return mdl.h1_2out == mdl.nn3.inputs[0]

@model.Constraint()
def connect_input3_2(mdl):
    return mdl.h2_2out == mdl.nn3.inputs[1]

@model.Constraint()
def connect_input3_3(mdl):
    return mdl.u3in == mdl.nn3.inputs[2]

@model.Constraint()
def connect_input3_4(mdl):
    return mdl.theta3  == mdl.nn3.inputs[3]

@model.Constraint()
def connect_input3_5(mdl):
    return mdl.qin3  == mdl.nn3.inputs[4]

@model.Constraint()
def connect_input3_6(mdl):
    return mdl.t1 == mdl.nn3.inputs[5]

@model.Constraint()
def connect_output3_1(mdl):
    return mdl.h1_3out == mdl.nn3.outputs[0]

@model.Constraint()
def connect_output3_2(mdl):
    return mdl.h2_3out == mdl.nn3.outputs[1]
###############################################
@model.Constraint()
def connect_input4_1(mdl):
    return mdl.h1_3out == mdl.nn4.inputs[0]

@model.Constraint()
def connect_input4_2(mdl):
    return mdl.h2_3out == mdl.nn4.inputs[1]

@model.Constraint()
def connect_input4_3(mdl):
    return mdl.u4in == mdl.nn4.inputs[2]

@model.Constraint()
def connect_input4_4(mdl):
    return mdl.theta4  == mdl.nn4.inputs[3]

@model.Constraint()
def connect_input4_5(mdl):
    return mdl.qin4  == mdl.nn4.inputs[4]

@model.Constraint()
def connect_input4_6(mdl):
    return mdl.t1 == mdl.nn4.inputs[5]

@model.Constraint()
def connect_output4_1(mdl):
    return mdl.h1_4out == mdl.nn4.outputs[0]

@model.Constraint()
def connect_output4_2(mdl):
    return mdl.h2_4out == mdl.nn4.outputs[1]
###############################################
@model.Constraint()
def connect_input5_1(mdl):
    return mdl.h1_4out == mdl.nn5.inputs[0]

@model.Constraint()
def connect_input5_2(mdl):
    return mdl.h2_4out == mdl.nn5.inputs[1]

@model.Constraint()
def connect_input5_3(mdl):
    return mdl.u5in == mdl.nn5.inputs[2]

@model.Constraint()
def connect_input5_4(mdl):
    return mdl.theta5  == mdl.nn5.inputs[3]

@model.Constraint()
def connect_input5_5(mdl):
    return mdl.qin5  == mdl.nn5.inputs[4]

@model.Constraint()
def connect_input5_6(mdl):
    return mdl.t1 == mdl.nn5.inputs[5]

@model.Constraint()
def connect_output5_1(mdl):
    return mdl.h1_5out == mdl.nn5.outputs[0]

@model.Constraint()
def connect_output5_2(mdl):
    return mdl.h2_5out == mdl.nn5.outputs[1]
###############################################
@model.Constraint()
def connect_input6_1(mdl):
    return mdl.h1_5out == mdl.nn6.inputs[0]

@model.Constraint()
def connect_input6_2(mdl):
    return mdl.h2_5out == mdl.nn6.inputs[1]

@model.Constraint()
def connect_input6_3(mdl):
    return mdl.u6in == mdl.nn6.inputs[2]

@model.Constraint()
def connect_input6_4(mdl):
    return mdl.theta6  == mdl.nn6.inputs[3]

@model.Constraint()
def connect_input6_5(mdl):
    return mdl.qin6  == mdl.nn6.inputs[4]

@model.Constraint()
def connect_input6_6(mdl):
    return mdl.t1 == mdl.nn6.inputs[5]

@model.Constraint()
def connect_output6_1(mdl):
    return mdl.h1_6out == mdl.nn6.outputs[0]

@model.Constraint()
def connect_output6_2(mdl):
    return mdl.h2_6out == mdl.nn6.outputs[1]
#########################################################################
model.Result=Var(initialize=0,within=Reals)
model.obj1 = Objective(expr=model.Result, sense=minimize)
#########################################################################
model.a1_1 = Constraint(expr = model.h1_1out-6-model.Result<=0)
model.a1_2 = Constraint(expr = model.h2_1out-6-model.Result<=0)
model.b1_1 = Constraint(expr = 4-model.h1_1out-model.Result<=0)
model.b1_2 = Constraint(expr = 4-model.h2_1out-model.Result<=0)
#########################################################################
model.a2_1 = Constraint(expr = model.h1_2out-6-model.Result<=0)
model.a2_2 = Constraint(expr = model.h2_2out-6-model.Result<=0)
model.b2_1 = Constraint(expr = 4-model.h1_2out-model.Result<=0)
model.b2_2 = Constraint(expr = 4-model.h2_2out-model.Result<=0)
#########################################################################
model.a3_1 = Constraint(expr = model.h1_3out-6-model.Result<=0)
model.a3_2 = Constraint(expr = model.h2_3out-6-model.Result<=0)
model.b3_1 = Constraint(expr = 4-model.h1_3out-model.Result<=0)
model.b3_2 = Constraint(expr = 4-model.h2_3out-model.Result<=0)
#########################################################################
model.a4_1 = Constraint(expr = model.h1_4out-6-model.Result<=0)
model.a4_2 = Constraint(expr = model.h2_4out-6-model.Result<=0)
model.b4_1 = Constraint(expr = 4-model.h1_4out-model.Result<=0)
model.b4_2 = Constraint(expr = 4-model.h2_4out-model.Result<=0)
#########################################################################
model.a5_1 = Constraint(expr = model.h1_5out-6-model.Result<=0)
model.a5_2 = Constraint(expr = model.h2_5out-6-model.Result<=0)
model.b5_1 = Constraint(expr = 4-model.h1_5out-model.Result<=0)
model.b5_2 = Constraint(expr = 4-model.h2_5out-model.Result<=0)
#########################################################################
model.a6_1 = Constraint(expr = model.h1_6out-6-model.Result<=0)
model.a6_2 = Constraint(expr = model.h2_6out-6-model.Result<=0)
model.b6_1 = Constraint(expr = 4-model.h1_6out-model.Result<=0)
model.b6_2 = Constraint(expr = 4-model.h2_6out-model.Result<=0)         

timeStart=time.time()
opt=SolverFactory('ipopt', executable='./ipopt')
results = opt.solve(model)
timeEnd=time.time()
slackVarValue=value(model.obj1)
print('Profit = ',slackVarValue,'time consumption = ',timeEnd-timeStart)