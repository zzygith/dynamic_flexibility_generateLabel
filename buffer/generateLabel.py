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

##############################################################################
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
for mmk in range(0,30):
    theta1=0.6-0.1+0.041
    theta2=0.7-0.1+0.041
    theta3=0.8-0.1+0.041
    theta4=0.9-0.1+0.041
    theta5=0.7-0.1+0.041
    theta6=0.6-0.1+0.041
    theta7=0.5-0.1+0.041
    theta8=0.3-0.1+0.041
    theta9=0.7-0.1+0.041
    theta10=0.6-0.1+0.041

    model = ConcreteModel()

    model.theta1=theta1
    model.theta2=theta2
    model.theta3=theta3
    model.theta4=theta4
    model.theta5=theta5
    model.theta6=theta6
    model.theta7=theta7
    model.theta8=theta8
    model.theta9=theta9
    model.theta10=theta10

    model.h1in=5.0
    model.t1 = 100.0
    model.t2 = 100.0
    model.t3 = 50.0
    model.t4 = 50.0
    model.t5 = 50.0
    model.t6 = 100.0
    model.t7 = 50.0
    model.t8 = 100.0
    model.t9 = 100.0
    model.t10 = 100.0


    def define_NN(par):
        LOC = """
model.nn%s= OmltBlock()
model.nn%s.build_formulation(formulation)
    """%(par,par)
        exec(LOC)

    for i in range(1,11):
        define_NN(i)


    def define_Var(par):
        LOC = """
model.h"""+str(par)+"""in=Var(initialize=5,within=Reals)
    """
        exec(LOC)

    def define_Control(par):
        LOC = """
model.u"""+str(par)+"""in=Var(initialize=0.1,within=Reals, bounds=(0, 5**0.5/10))
    """
        exec(LOC)

    def define_Constraint(par):
        LOC = """
@model.Constraint()
def connect_input%s_1(mdl):
    return mdl.h%sin == mdl.nn%s.inputs[0]

@model.Constraint()
def connect_input%s_2(mdl):
    return mdl.u%sin == mdl.nn%s.inputs[1]

@model.Constraint()
def connect_input%s_3(mdl):
    return mdl.theta%s  == mdl.nn%s.inputs[2]

@model.Constraint()
def connect_input%s_4(mdl):
    return mdl.t%s == mdl.nn%s.inputs[3]

@model.Constraint()
def connect_output%s_1(mdl):
    return mdl.h%sin == mdl.nn%s.outputs[0]

    """%(par,par,par,par,par,par,par,par,par,par,par,par,par,par+1,par)
        exec(LOC)

    for i in range(2,12):
        define_Var(i)

    for i in range(1,11):
        define_Control(i)

    for i in range(1,11):
        define_Constraint(i)

    model.Result=Var(initialize=0,within=Reals)

    model.obj1 = Objective(expr=model.Result, sense=minimize)
    # model.e1 = Constraint(expr = model.h1==model.u1*theta1)
    # model.e2 = Constraint(expr = model.h2==model.u2*theta2+model.h1)
    # model.j1 = Constraint(expr = model.h1in-10-model.Result<=0)
    # model.j2 = Constraint(expr = model.h2in-10-model.Result<=0)
    # model.j3 = Constraint(expr = model.h3in-10-model.Result<=0)
    # model.j4 = Constraint(expr = model.h4in-10-model.Result<=0)
    # model.j5 = Constraint(expr = model.h5in-10-model.Result<=0)
    # model.j6 = Constraint(expr = model.h6in-10-model.Result<=0)
    # model.j7 = Constraint(expr = model.h7in-10-model.Result<=0)
    # model.j8 = Constraint(expr = model.h8in-10-model.Result<=0)
    # model.j9 = Constraint(expr = model.h9in-10-model.Result<=0)
    # model.j10 = Constraint(expr = model.h10in-10-model.Result<=0)

    def define_aConstraints(par):
        LOC = """
model.a%s = Constraint(expr = model.h%sin-10-model.Result<=0)
    """%(par,par)
        exec(LOC)
    for i in range(2,12):
        define_aConstraints(i)

    def define_bConstraints(par):
        LOC = """
model.b%s = Constraint(expr = 1-model.h%sin-model.Result<=0)
    """%(par,par)
        exec(LOC)
    for i in range(2,12):
        define_bConstraints(i)

    timeStart=time.time()
    opt=SolverFactory('ipopt', executable='./ipopt')
    results = opt.solve(model)
    timeEnd=time.time()
    print('Profit = ',value(model.obj1))
    print('time consumption = ',timeEnd-timeStart)