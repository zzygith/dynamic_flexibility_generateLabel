import onnx
from omlt.io import write_onnx_model_with_bounds,load_onnx_neural_network
from omlt import OmltBlock, OffsetScaling
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
from pyomo.environ import *
import time

pytorch_model='./buffer_pinnc.onnx'
onnx_model = onnx.load(pytorch_model)
network_definition = load_onnx_neural_network(onnx_model)
for layer_id, layer in enumerate(network_definition.layers):
    print(f"{layer_id}\t{layer}\t{layer.activation}")
formulation = FullSpaceNNFormulation(network_definition)

###########################################################################################
###########################################################################################

# theta1=0.6-0.1+0.039
# theta2=0.7-0.1+0.039
# theta3=0.8-0.1+0.039
# theta4=0.9-0.1+0.039
# theta5=0.7-0.1+0.039
# theta6=0.6-0.1+0.039
# theta7=0.5-0.1+0.039
# theta8=0.3-0.1+0.039
# theta9=0.7-0.1+0.039
# theta10=0.6-0.1+0.039

# model = ConcreteModel()

# model.theta1=theta1
# model.theta2=theta2
# model.theta3=theta3
# model.theta4=theta4
# model.theta5=theta5
# model.theta6=theta6
# model.theta7=theta7
# model.theta8=theta8
# model.theta9=theta9
# model.theta10=theta10

# model.h1in=5.0
# model.t1 = 100.0
# model.t2 = 100.0
# model.t3 = 50.0
# model.t4 = 50.0
# model.t5 = 50.0
# model.t6 = 100.0
# model.t7 = 50.0
# model.t8 = 100.0
# model.t9 = 100.0
# model.t10 = 100.0


# def define_NN(par):
#     LOC = """
# model.nn%s= OmltBlock()
# model.nn%s.build_formulation(formulation)
# """%(par,par)
#     exec(LOC)

# for i in range(1,11):
#     define_NN(i)


# def define_Var(par):
#     LOC = """
# model.h"""+str(par)+"""in=Var(initialize=5,within=Reals)
# """
#     exec(LOC)

# def define_Control(par):
#     LOC = """
# model.u"""+str(par)+"""in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
# """
#     exec(LOC)

# def define_Constraint(par):
#     LOC = """
# @model.Constraint()
# def connect_input%s_1(mdl):
#     return mdl.h%sin == mdl.nn%s.inputs[0]

# @model.Constraint()
# def connect_input%s_2(mdl):
#     return mdl.u%sin == mdl.nn%s.inputs[1]

# @model.Constraint()
# def connect_input%s_3(mdl):
#     return mdl.theta%s  == mdl.nn%s.inputs[2]

# @model.Constraint()
# def connect_input%s_4(mdl):
#     return mdl.t%s == mdl.nn%s.inputs[3]

# @model.Constraint()
# def connect_output%s_1(mdl):
#     return mdl.h%sin == mdl.nn%s.outputs[0]

# """%(par,par,par,par,par,par,par,par,par,par,par,par,par,par+1,par)
#     exec(LOC)

# for i in range(2,12):
#     define_Var(i)

# for i in range(1,11):
#     define_Control(i)

# for i in range(1,11):
#     define_Constraint(i)

# model.Result=Var(initialize=0,within=Reals)

# model.obj1 = Objective(expr=model.Result, sense=minimize)
# # model.e1 = Constraint(expr = model.h1==model.u1*theta1)
# # model.e2 = Constraint(expr = model.h2==model.u2*theta2+model.h1)
# # model.j1 = Constraint(expr = model.h1in-10-model.Result<=0)
# # model.j2 = Constraint(expr = model.h2in-10-model.Result<=0)
# # model.j3 = Constraint(expr = model.h3in-10-model.Result<=0)
# # model.j4 = Constraint(expr = model.h4in-10-model.Result<=0)
# # model.j5 = Constraint(expr = model.h5in-10-model.Result<=0)
# # model.j6 = Constraint(expr = model.h6in-10-model.Result<=0)
# # model.j7 = Constraint(expr = model.h7in-10-model.Result<=0)
# # model.j8 = Constraint(expr = model.h8in-10-model.Result<=0)
# # model.j9 = Constraint(expr = model.h9in-10-model.Result<=0)
# # model.j10 = Constraint(expr = model.h10in-10-model.Result<=0)

# def define_aConstraints(par):
#     LOC = """
# model.a%s = Constraint(expr = model.h%sin-10-model.Result<=0)
# """%(par,par)
#     exec(LOC)
# for i in range(2,12):
#     define_aConstraints(i)

# def define_bConstraints(par):
#     LOC = """
# model.b%s = Constraint(expr = 1-model.h%sin-model.Result<=0)
# """%(par,par)
#     exec(LOC)
# for i in range(2,12):
#     define_bConstraints(i)

# timeStart=time.time()
# opt=SolverFactory('ipopt', executable='./ipopt')
# results = opt.solve(model)
# timeEnd=time.time()
# print('Profit = ',value(model.obj1))
# print('time consumption = ',timeEnd-timeStart)

###########################################################################################
###########################################################################################

# for mmk in range(0,30):
#     theta1=0.6-0.1+0.041
#     theta2=0.7-0.1+0.041
#     theta3=0.8-0.1+0.041
#     theta4=0.9-0.1+0.041
#     theta5=0.7-0.1+0.041
#     theta6=0.6-0.1+0.041
#     theta7=0.5-0.1+0.041
#     theta8=0.3-0.1+0.041
#     theta9=0.7-0.1+0.041
#     theta10=0.6-0.1+0.041

#     model = ConcreteModel()

#     model.theta1=theta1
#     model.theta2=theta2
#     model.theta3=theta3
#     model.theta4=theta4
#     model.theta5=theta5
#     model.theta6=theta6
#     model.theta7=theta7
#     model.theta8=theta8
#     model.theta9=theta9
#     model.theta10=theta10

#     model.h1in=5.0
#     model.t1 = 100.0
#     model.t2 = 100.0
#     model.t3 = 50.0
#     model.t4 = 50.0
#     model.t5 = 50.0
#     model.t6 = 100.0
#     model.t7 = 50.0
#     model.t8 = 100.0
#     model.t9 = 100.0
#     model.t10 = 100.0


#     def define_NN(par):
#         LOC = """
# model.nn%s= OmltBlock()
# model.nn%s.build_formulation(formulation)
#     """%(par,par)
#         exec(LOC)

#     for i in range(1,11):
#         define_NN(i)


#     def define_Var(par):
#         LOC = """
# model.h"""+str(par)+"""in=Var(initialize=5,within=Reals)
#     """
#         exec(LOC)

#     def define_Control(par):
#         LOC = """
# model.u"""+str(par)+"""in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
#     """
#         exec(LOC)

#     def define_Constraint(par):
#         LOC = """
# @model.Constraint()
# def connect_input%s_1(mdl):
#     return mdl.h%sin == mdl.nn%s.inputs[0]

# @model.Constraint()
# def connect_input%s_2(mdl):
#     return mdl.u%sin == mdl.nn%s.inputs[1]

# @model.Constraint()
# def connect_input%s_3(mdl):
#     return mdl.theta%s  == mdl.nn%s.inputs[2]

# @model.Constraint()
# def connect_input%s_4(mdl):
#     return mdl.t%s == mdl.nn%s.inputs[3]

# @model.Constraint()
# def connect_output%s_1(mdl):
#     return mdl.h%sin == mdl.nn%s.outputs[0]

#     """%(par,par,par,par,par,par,par,par,par,par,par,par,par,par+1,par)
#         exec(LOC)

#     for i in range(2,12):
#         define_Var(i)

#     for i in range(1,11):
#         define_Control(i)

#     for i in range(1,11):
#         define_Constraint(i)

#     model.Result=Var(initialize=0,within=Reals)

#     model.obj1 = Objective(expr=model.Result, sense=minimize)
#     # model.e1 = Constraint(expr = model.h1==model.u1*theta1)
#     # model.e2 = Constraint(expr = model.h2==model.u2*theta2+model.h1)
#     # model.j1 = Constraint(expr = model.h1in-10-model.Result<=0)
#     # model.j2 = Constraint(expr = model.h2in-10-model.Result<=0)
#     # model.j3 = Constraint(expr = model.h3in-10-model.Result<=0)
#     # model.j4 = Constraint(expr = model.h4in-10-model.Result<=0)
#     # model.j5 = Constraint(expr = model.h5in-10-model.Result<=0)
#     # model.j6 = Constraint(expr = model.h6in-10-model.Result<=0)
#     # model.j7 = Constraint(expr = model.h7in-10-model.Result<=0)
#     # model.j8 = Constraint(expr = model.h8in-10-model.Result<=0)
#     # model.j9 = Constraint(expr = model.h9in-10-model.Result<=0)
#     # model.j10 = Constraint(expr = model.h10in-10-model.Result<=0)

#     def define_aConstraints(par):
#         LOC = """
# model.a%s = Constraint(expr = model.h%sin-10-model.Result<=0)
#     """%(par,par)
#         exec(LOC)
#     for i in range(2,12):
#         define_aConstraints(i)

#     def define_bConstraints(par):
#         LOC = """
# model.b%s = Constraint(expr = 1-model.h%sin-model.Result<=0)
#     """%(par,par)
#         exec(LOC)
#     for i in range(2,12):
#         define_bConstraints(i)

#     timeStart=time.time()
#     opt=SolverFactory('ipopt', executable='./ipopt')
#     results = opt.solve(model)
#     timeEnd=time.time()
#     print('Profit = ',value(model.obj1))
#     print('time consumption = ',timeEnd-timeStart)

###########################################################################################
###########################################################################################
## 不用‘’写， 只计算间隔尾部

# for mmk in range(0,30):
#     theta1=0.5
#     theta2=0.5
#     theta3=0.5
#     theta4=0.5

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
#     print('Profit = ',slackVarValue,'time consumption = ',timeEnd-timeStart)


###########################################################################################
###########################################################################################
## 不用‘’写，在间隔内部取点验证

for mmk in range(0,30):
    theta1=0.5
    theta2=0.5
    theta3=0.5
    theta4=0.5

    model = ConcreteModel()

    model.theta1=theta1
    model.theta2=theta2
    model.theta3=theta3
    model.theta4=theta4


    model.t1 = 100.0
    model.t2 = 100.0
    model.t3 = 100.0
    model.t4 = 100.0
    model.interT1 = 50.0
    model.interT2 = 40.0
    model.interT3 = 60.0
    model.interT4 = 80.0

    model.nn1= OmltBlock()
    model.nn1.build_formulation(formulation)

    model.nn1_1= OmltBlock()
    model.nn1_1.build_formulation(formulation)

    model.nn2= OmltBlock()
    model.nn2.build_formulation(formulation)

    model.nn2_1= OmltBlock()
    model.nn2_1.build_formulation(formulation)

    model.nn3= OmltBlock()
    model.nn3.build_formulation(formulation)

    model.nn3_1= OmltBlock()
    model.nn3_1.build_formulation(formulation)

    model.nn4= OmltBlock()
    model.nn4.build_formulation(formulation)

    model.nn4_1= OmltBlock()
    model.nn4_1.build_formulation(formulation)

    model.h1in=5.0
    model.h2in=Var(initialize=5,within=Reals)
    model.h3in=Var(initialize=5,within=Reals)
    model.h4in=Var(initialize=5,within=Reals)
    model.h5in=Var(initialize=5,within=Reals)

    model.h1_1=Var(initialize=5,within=Reals)
    model.h2_1=Var(initialize=5,within=Reals)
    model.h3_1=Var(initialize=5,within=Reals)
    model.h4_1=Var(initialize=5,within=Reals)



    model.u1in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
    model.u2in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
    model.u3in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
    model.u4in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))


    #########################################################################
    @model.Constraint()
    def connect_input1_1(mdl):
        return mdl.h1in == mdl.nn1.inputs[0]

    @model.Constraint()
    def connect_input1_2(mdl):
        return mdl.u1in == mdl.nn1.inputs[1]

    @model.Constraint()
    def connect_input1_3(mdl):
        return mdl.theta1  == mdl.nn1.inputs[2]

    @model.Constraint()
    def connect_input1_4(mdl):
        return mdl.t1 == mdl.nn1.inputs[3]

    @model.Constraint()
    def connect_output1_1(mdl):
        return mdl.h2in == mdl.nn1.outputs[0]
    #################################################
    @model.Constraint()
    def connect_input1_1_1(mdl):
        return mdl.h1in == mdl.nn1_1.inputs[0]

    @model.Constraint()
    def connect_input1_1_2(mdl):
        return mdl.u1in == mdl.nn1_1.inputs[1]

    @model.Constraint()
    def connect_input1_1_3(mdl):
        return mdl.theta1  == mdl.nn1_1.inputs[2]

    @model.Constraint()
    def connect_input1_1_4(mdl):
        return mdl.interT1 == mdl.nn1_1.inputs[3]

    @model.Constraint()
    def connect_output1_1_1(mdl):
        return mdl.h1_1 == mdl.nn1_1.outputs[0]

    ###############################################
    @model.Constraint()
    def connect_input2_1(mdl):
        return mdl.h2in == mdl.nn2.inputs[0]

    @model.Constraint()
    def connect_input2_2(mdl):
        return mdl.u2in == mdl.nn2.inputs[1]

    @model.Constraint()
    def connect_input2_3(mdl):
        return mdl.theta2  == mdl.nn2.inputs[2]

    @model.Constraint()
    def connect_input2_4(mdl):
        return mdl.t2 == mdl.nn2.inputs[3]

    @model.Constraint()
    def connect_output2_1(mdl):
        return mdl.h3in == mdl.nn2.outputs[0]
    #######################################################
    @model.Constraint()
    def connect_input2_1_1(mdl):
        return mdl.h2in == mdl.nn2_1.inputs[0]

    @model.Constraint()
    def connect_input2_1_2(mdl):
        return mdl.u2in == mdl.nn2_1.inputs[1]

    @model.Constraint()
    def connect_input2_1_3(mdl):
        return mdl.theta2  == mdl.nn2_1.inputs[2]

    @model.Constraint()
    def connect_input2_1_4(mdl):
        return mdl.interT2 == mdl.nn2_1.inputs[3]

    @model.Constraint()
    def connect_output2_1_1(mdl):
        return mdl.h2_1 == mdl.nn2_1.outputs[0]

    ###############################################
    @model.Constraint()
    def connect_input3_1(mdl):
        return mdl.h3in == mdl.nn3.inputs[0]

    @model.Constraint()
    def connect_input3_2(mdl):
        return mdl.u3in == mdl.nn3.inputs[1]

    @model.Constraint()
    def connect_input3_3(mdl):
        return mdl.theta3  == mdl.nn3.inputs[2]

    @model.Constraint()
    def connect_input3_4(mdl):
        return mdl.t3 == mdl.nn3.inputs[3]

    @model.Constraint()
    def connect_output3_1(mdl):
        return mdl.h4in == mdl.nn3.outputs[0]
    
    ###############################################
    @model.Constraint()
    def connect_input3_1_1(mdl):
        return mdl.h3in == mdl.nn3_1.inputs[0]

    @model.Constraint()
    def connect_input3_1_2(mdl):
        return mdl.u3in == mdl.nn3_1.inputs[1]

    @model.Constraint()
    def connect_input3_1_3(mdl):
        return mdl.theta3  == mdl.nn3_1.inputs[2]

    @model.Constraint()
    def connect_input3_1_4(mdl):
        return mdl.interT3 == mdl.nn3_1.inputs[3]

    @model.Constraint()
    def connect_output3_1_1(mdl):
        return mdl.h3_1 == mdl.nn3_1.outputs[0]

    ###############################################
    @model.Constraint()
    def connect_input4_1(mdl):
        return mdl.h4in == mdl.nn4.inputs[0]

    @model.Constraint()
    def connect_input4_2(mdl):
        return mdl.u4in == mdl.nn4.inputs[1]

    @model.Constraint()
    def connect_input4_3(mdl):
        return mdl.theta4  == mdl.nn4.inputs[2]

    @model.Constraint()
    def connect_input4_4(mdl):
        return mdl.t4 == mdl.nn4.inputs[3]

    @model.Constraint()
    def connect_output4_1(mdl):
        return mdl.h5in == mdl.nn4.outputs[0]

    ###############################################
    @model.Constraint()
    def connect_input4_1_1(mdl):
        return mdl.h4in == mdl.nn4_1.inputs[0]

    @model.Constraint()
    def connect_input4_1_2(mdl):
        return mdl.u4in == mdl.nn4_1.inputs[1]

    @model.Constraint()
    def connect_input4_1_3(mdl):
        return mdl.theta4  == mdl.nn4_1.inputs[2]

    @model.Constraint()
    def connect_input4_1_4(mdl):
        return mdl.interT4 == mdl.nn4_1.inputs[3]

    @model.Constraint()
    def connect_output4_1_1(mdl):
        return mdl.h4_1 == mdl.nn4_1.outputs[0]

    ###############################################
    model.Result=Var(initialize=0,within=Reals)

    model.obj1 = Objective(expr=model.Result, sense=minimize)


    model.a2 = Constraint(expr = model.h2in-10-model.Result<=0)
    model.a3 = Constraint(expr = model.h3in-10-model.Result<=0)
    model.a4 = Constraint(expr = model.h4in-10-model.Result<=0)
    model.a5 = Constraint(expr = model.h5in-10-model.Result<=0)
    model.a2_1 = Constraint(expr = model.h1_1-10-model.Result<=0)
    model.a3_1 = Constraint(expr = model.h2_1-10-model.Result<=0)
    model.a4_1 = Constraint(expr = model.h3_1-10-model.Result<=0)
    model.a5_1 = Constraint(expr = model.h4_1-10-model.Result<=0)


    model.b2 = Constraint(expr = 1-model.h2in-model.Result<=0)
    model.b3 = Constraint(expr = 1-model.h3in-model.Result<=0)
    model.b4 = Constraint(expr = 1-model.h4in-model.Result<=0)
    model.b5 = Constraint(expr = 1-model.h5in-model.Result<=0)
    model.b2_1 = Constraint(expr = 1-model.h1_1-model.Result<=0)
    model.b3_1 = Constraint(expr = 1-model.h2_1-model.Result<=0)
    model.b4_1 = Constraint(expr = 1-model.h3_1-model.Result<=0)
    model.b5_1 = Constraint(expr = 1-model.h4_1-model.Result<=0)

    timeStart=time.time()
    opt=SolverFactory('ipopt', executable='./ipopt')
    results = opt.solve(model)
    timeEnd=time.time()
    slackVarValue=value(model.obj1)
    print('Profit = ',slackVarValue,'time consumption = ',timeEnd-timeStart)