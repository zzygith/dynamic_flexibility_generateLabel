# import onnx
# from omlt.io import write_onnx_model_with_bounds,load_onnx_neural_network
# from omlt import OmltBlock, OffsetScaling
# from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
# from pyomo.environ import *
# #import time
# import os, time, random

# pytorch_model='./buffer_pinnc.onnx'
# onnx_model = onnx.load(pytorch_model)
# network_definition = load_onnx_neural_network(onnx_model)
# for layer_id, layer in enumerate(network_definition.layers):
#     print(f"{layer_id}\t{layer}\t{layer.activation}")
# formulation = FullSpaceNNFormulation(network_definition)

# # def calculateFunc(uncertainParameterList):
# #     # theta1=uncertainParameterList[0]
# #     # theta2=uncertainParameterList[1]
# #     # theta3=uncertainParameterList[2]
# #     # theta4=uncertainParameterList[3]
# #     # theta5=uncertainParameterList[4]
# #     # theta6=uncertainParameterList[5]
# #     # theta7=uncertainParameterList[6]
# #     # theta8=uncertainParameterList[7]
# #     # theta9=uncertainParameterList[8]
# #     # theta10=uncertainParameterList[9]

# #     theta1=uncertainParameterList
# #     theta2=uncertainParameterList
# #     theta3=uncertainParameterList
# #     theta4=uncertainParameterList
# #     theta5=uncertainParameterList
# #     theta6=uncertainParameterList
# #     theta7=uncertainParameterList
# #     theta8=uncertainParameterList
# #     theta9=uncertainParameterList
# #     theta10=uncertainParameterList

# #     model = ConcreteModel()

# #     model.theta1=theta1
# #     model.theta2=theta2
# #     model.theta3=theta3
# #     model.theta4=theta4
# #     model.theta5=theta5
# #     model.theta6=theta6
# #     model.theta7=theta7
# #     model.theta8=theta8
# #     model.theta9=theta9
# #     model.theta10=theta10

# #     model.h1in=5.0
# #     model.t1 = 100.0
# #     model.t2 = 100.0
# #     model.t3 = 50.0
# #     model.t4 = 50.0
# #     model.t5 = 50.0
# #     model.t6 = 100.0
# #     model.t7 = 50.0
# #     model.t8 = 100.0
# #     model.t9 = 100.0
# #     model.t10 = 100.0


# #     def define_NN(par):
# #         LOC = """
# # model.nn%s= OmltBlock()
# # model.nn%s.build_formulation(formulation)
# #     """%(par,par)
# #         exec(LOC)

# #     for i in range(1,11):
# #         define_NN(i)


# #     def define_Var(par):
# #         LOC = """
# # model.h"""+str(par)+"""in=Var(initialize=5,within=Reals)
# #     """
# #         exec(LOC)

# #     def define_Control(par):
# #         LOC = """
# # model.u"""+str(par)+"""in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
# #     """
# #         exec(LOC)

# #     def define_Constraint(par):
# #         LOC = """
# # @model.Constraint()
# # def connect_input%s_1(mdl):
# #     return mdl.h%sin == mdl.nn%s.inputs[0]

# # @model.Constraint()
# # def connect_input%s_2(mdl):
# #     return mdl.u%sin == mdl.nn%s.inputs[1]

# # @model.Constraint()
# # def connect_input%s_3(mdl):
# #     return mdl.theta%s  == mdl.nn%s.inputs[2]

# # @model.Constraint()
# # def connect_input%s_4(mdl):
# #     return mdl.t%s == mdl.nn%s.inputs[3]

# # @model.Constraint()
# # def connect_output%s_1(mdl):
# #     return mdl.h%sin == mdl.nn%s.outputs[0]

# #     """%(par,par,par,par,par,par,par,par,par,par,par,par,par,par+1,par)
# #         exec(LOC)

# #     for i in range(2,12):
# #         define_Var(i)

# #     for i in range(1,11):
# #         define_Control(i)

# #     for i in range(1,11):
# #         define_Constraint(i)

# #     model.Result=Var(initialize=0,within=Reals)

# #     model.obj1 = Objective(expr=model.Result, sense=minimize)

# #     def define_aConstraints(par):
# #         LOC = """
# # model.a%s = Constraint(expr = model.h%sin-10-model.Result<=0)
# #     """%(par,par)
# #         exec(LOC)
# #     for i in range(2,12):
# #         define_aConstraints(i)

# #     def define_bConstraints(par):
# #         LOC = """
# # model.b%s = Constraint(expr = 1-model.h%sin-model.Result<=0)
# #     """%(par,par)
# #         exec(LOC)
# #     for i in range(2,12):
# #         define_bConstraints(i)

# #     timeStart=time.time()
# #     opt=SolverFactory('ipopt', executable='./ipopt')
# #     results = opt.solve(model)
# #     timeEnd=time.time()
# #     print('Profit = ',value(model.obj1))
# #     print('time consumption = ',timeEnd-timeStart)



# ##############################################################################
# def calculateFunc(uncertainParameterList):
#     theta1=uncertainParameterList[0]
#     theta2=uncertainParameterList[1]
#     theta3=uncertainParameterList[2]
#     theta4=uncertainParameterList[3]

#     model = ConcreteModel()

#     model.theta1=theta1
#     model.theta2=theta2
#     model.theta3=theta3
#     model.theta4=theta4


#     model.t1 = 100.0
#     model.t2 = 100.0
#     model.t3 = 100.0
#     model.t4 = 100.0

#     model.nn1= OmltBlock()
#     model.nn1.build_formulation(formulation)

#     model.nn2= OmltBlock()
#     model.nn2.build_formulation(formulation)

#     model.nn3= OmltBlock()
#     model.nn3.build_formulation(formulation)

#     model.nn4= OmltBlock()
#     model.nn4.build_formulation(formulation)

#     model.h1in=5.0
#     model.h2in=Var(initialize=5,within=Reals)
#     model.h3in=Var(initialize=5,within=Reals)
#     model.h4in=Var(initialize=5,within=Reals)
#     model.h5in=Var(initialize=5,within=Reals)


#     model.u1in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
#     model.u2in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
#     model.u3in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
#     model.u4in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))


#     #########################################################################
#     @model.Constraint()
#     def connect_input1_1(mdl):
#         return mdl.h1in == mdl.nn1.inputs[0]

#     @model.Constraint()
#     def connect_input1_2(mdl):
#         return mdl.u1in == mdl.nn1.inputs[1]

#     @model.Constraint()
#     def connect_input1_3(mdl):
#         return mdl.theta1  == mdl.nn1.inputs[2]

#     @model.Constraint()
#     def connect_input1_4(mdl):
#         return mdl.t1 == mdl.nn1.inputs[3]

#     @model.Constraint()
#     def connect_output1_1(mdl):
#         return mdl.h2in == mdl.nn1.outputs[0]

#     ###############################################
#     @model.Constraint()
#     def connect_input2_1(mdl):
#         return mdl.h2in == mdl.nn2.inputs[0]

#     @model.Constraint()
#     def connect_input2_2(mdl):
#         return mdl.u2in == mdl.nn2.inputs[1]

#     @model.Constraint()
#     def connect_input2_3(mdl):
#         return mdl.theta2  == mdl.nn2.inputs[2]

#     @model.Constraint()
#     def connect_input2_4(mdl):
#         return mdl.t2 == mdl.nn2.inputs[3]

#     @model.Constraint()
#     def connect_output2_1(mdl):
#         return mdl.h3in == mdl.nn2.outputs[0]

#     ###############################################
#     @model.Constraint()
#     def connect_input3_1(mdl):
#         return mdl.h3in == mdl.nn3.inputs[0]

#     @model.Constraint()
#     def connect_input3_2(mdl):
#         return mdl.u3in == mdl.nn3.inputs[1]

#     @model.Constraint()
#     def connect_input3_3(mdl):
#         return mdl.theta3  == mdl.nn3.inputs[2]

#     @model.Constraint()
#     def connect_input3_4(mdl):
#         return mdl.t3 == mdl.nn3.inputs[3]

#     @model.Constraint()
#     def connect_output3_1(mdl):
#         return mdl.h4in == mdl.nn3.outputs[0]

#     ###############################################
#     @model.Constraint()
#     def connect_input4_1(mdl):
#         return mdl.h4in == mdl.nn4.inputs[0]

#     @model.Constraint()
#     def connect_input4_2(mdl):
#         return mdl.u4in == mdl.nn4.inputs[1]

#     @model.Constraint()
#     def connect_input4_3(mdl):
#         return mdl.theta4  == mdl.nn4.inputs[2]

#     @model.Constraint()
#     def connect_input4_4(mdl):
#         return mdl.t4 == mdl.nn4.inputs[3]

#     @model.Constraint()
#     def connect_output4_1(mdl):
#         return mdl.h5in == mdl.nn4.outputs[0]

#     ###############################################


#     #########################################################################
#     model.Result=Var(initialize=0,within=Reals)

#     model.obj1 = Objective(expr=model.Result, sense=minimize)


#     model.a2 = Constraint(expr = model.h2in-10-model.Result<=0)
#     model.a3 = Constraint(expr = model.h3in-10-model.Result<=0)
#     model.a4 = Constraint(expr = model.h4in-10-model.Result<=0)
#     model.a5 = Constraint(expr = model.h5in-10-model.Result<=0)


#     model.b2 = Constraint(expr = 1-model.h2in-model.Result<=0)
#     model.b3 = Constraint(expr = 1-model.h3in-model.Result<=0)
#     model.b4 = Constraint(expr = 1-model.h4in-model.Result<=0)
#     model.b5 = Constraint(expr = 1-model.h5in-model.Result<=0)

#     timeStart=time.time()
#     opt=SolverFactory('ipopt', executable='./ipopt')
#     results = opt.solve(model)
#     timeEnd=time.time()
#     slackVarValue=value(model.obj1)
#     print(uncertainParameterList,'Profit = ',slackVarValue,'time consumption = ',timeEnd-timeStart)
#     if slackVarValue<=0:
#         flag=1
#     else:
#         flag=0
#     return ([theta1,theta2,theta3,theta4,flag])
#     #print('time consumption = ',timeEnd-timeStart)


# def labelFunc(uncertainParameterList):
#     print('Run task (%s)...' % (os.getpid()))
#     singleResult=calculateFunc(uncertainParameterList)
#     print('Finish task (%s)...' % (os.getpid()))
#     return singleResult



#####################################################################################
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
    upperBound=7
    lowerBound=3
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

    if qin1-20*(theta1-0.3)**2-0.3<=0 and qin2-20*(theta2-0.3)**2-0.3<=0 and qin3-20*(theta3-0.3)**2-0.3<=0 and qin4-20*(theta4-0.3)**2-0.3<=0 and qin5-20*(theta5-0.3)**2-0.3<=0 and qin6-20*(theta6-0.3)**2-0.3<=0 :
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
        model.a1_1 = Constraint(expr = model.h1_1out-upperBound-model.Result<=0)
        model.a1_2 = Constraint(expr = model.h2_1out-upperBound-model.Result<=0)
        model.b1_1 = Constraint(expr = lowerBound-model.h1_1out-model.Result<=0)
        model.b1_2 = Constraint(expr = lowerBound-model.h2_1out-model.Result<=0)
        #########################################################################
        model.a2_1 = Constraint(expr = model.h1_2out-upperBound-model.Result<=0)
        model.a2_2 = Constraint(expr = model.h2_2out-upperBound-model.Result<=0)
        model.b2_1 = Constraint(expr = lowerBound-model.h1_2out-model.Result<=0)
        model.b2_2 = Constraint(expr = lowerBound-model.h2_2out-model.Result<=0)
        #########################################################################
        model.a3_1 = Constraint(expr = model.h1_3out-upperBound-model.Result<=0)
        model.a3_2 = Constraint(expr = model.h2_3out-upperBound-model.Result<=0)
        model.b3_1 = Constraint(expr = lowerBound-model.h1_3out-model.Result<=0)
        model.b3_2 = Constraint(expr = lowerBound-model.h2_3out-model.Result<=0)
        #########################################################################
        model.a4_1 = Constraint(expr = model.h1_4out-upperBound-model.Result<=0)
        model.a4_2 = Constraint(expr = model.h2_4out-upperBound-model.Result<=0)
        model.b4_1 = Constraint(expr = lowerBound-model.h1_4out-model.Result<=0)
        model.b4_2 = Constraint(expr = lowerBound-model.h2_4out-model.Result<=0)
        #########################################################################
        model.a5_1 = Constraint(expr = model.h1_5out-upperBound-model.Result<=0)
        model.a5_2 = Constraint(expr = model.h2_5out-upperBound-model.Result<=0)
        model.b5_1 = Constraint(expr = lowerBound-model.h1_5out-model.Result<=0)
        model.b5_2 = Constraint(expr = lowerBound-model.h2_5out-model.Result<=0)
        #########################################################################
        model.a6_1 = Constraint(expr = model.h1_6out-upperBound-model.Result<=0)
        model.a6_2 = Constraint(expr = model.h2_6out-upperBound-model.Result<=0)
        model.b6_1 = Constraint(expr = lowerBound-model.h1_6out-model.Result<=0)
        model.b6_2 = Constraint(expr = lowerBound-model.h2_6out-model.Result<=0)         
        
        #timeStart=time.time()
        opt=SolverFactory('ipopt', executable='./ipopt')
        results = opt.solve(model)
        #timeEnd=time.time()
        slackVarValue=value(model.obj1)
        #print(uncertainParameterList,'Profit = ',slackVarValue,'time consumption = ',timeEnd-timeStart)
        if slackVarValue<=0:
            flag=1
        else:
            flag=0
        return ([theta1,qin1,theta2,qin2,theta3,qin3,theta4,qin4,theta5,qin5,theta6,qin6,flag,flag])
    else:
        return ([theta1,qin1,theta2,qin2,theta3,qin3,theta4,qin4,theta5,qin5,theta6,qin6,0,0])


def labelFunc(uncertainParameterList):
    print('Run task (%s)...' % (os.getpid()))
    singleResult=calculateFunc(uncertainParameterList)
    print('Finish task (%s)...' % (os.getpid()))
    return singleResult