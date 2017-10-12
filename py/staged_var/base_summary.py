import math
import tensorflow as tf


default_summary_collection="default_summary_collection"
default_summary_collection_gradient="default_summary_collection_gradient"

def f_summary_one(var,name=None,scopename=None,methods=None):
    if name is None:
        name=var.name.replace(":","_")
    else:
        name=name.replace(":","_")
    print("tensor into summary: ",var)
    print("tag name: ",name)

    if scopename is None:
        if name is None:
            scopename = var.name.replace(":","_") + "/summary"
        else:
            scopename = name.replace(":","_") + "/summary"
    else:
        scopename=scopename

    if methods is None:
        method_set ={"histogram":{},
                      "scalar":{"max","min","mean","std_dev","abs_min","abs_mean","num_zero"}}
    else:
        method_set = methods
    with tf.name_scope(scopename):
        if method_set.has_key("histogram"):
            tf.summary.histogram(name + "/histogram/", var)
        if method_set.has_key("scalar"):
            if "max" in method_set["scalar"]:
                var_max=tf.reduce_max(var)
                tf.summary.scalar(name + "/max/", var_max)
            if "min" in method_set["scalar"]:
                var_min=tf.reduce_min(var)
                tf.summary.scalar(name + "/min/", var_min)
            if "mean" in method_set["scalar"]:
                var_mean=tf.reduce_mean(var)
                tf.summary.scalar(name + "/mean/", var_mean)
                if "std_dev" in method_set["scalar"]:
                    var_std_dev=tf.sqrt(tf.reduce_mean(tf.square(var-var_mean)))
                    tf.summary.scalar(name + "/std_dev/", var_std_dev)
            elif "std_dev" in method_set["scalar"]:
                var_mean = tf.reduce_mean(var)
                var_std_dev = tf.sqrt(tf.reduce_mean(tf.square(var - var_mean)))
                tf.summary.scalar(name + "/std_dev/", var_std_dev)
            if "abs_min" in method_set["scalar"]:
                var_abs_min=tf.reduce_min(tf.abs(var))
                tf.summary.scalar(name + "/abs_min/", var_abs_min)
            if "abs_mean" in method_set["scalar"]:
                var_abs_mean=tf.reduce_mean(tf.abs(var))
                tf.summary.scalar(name + "/abs_mean/", var_abs_mean)
            if "num_zero" in method_set["scalar"]:
                x1=tf.equal(var,0)
                x2=tf.to_int32(x1)
                var_num_zero=tf.reduce_sum(x2)
                tf.summary.scalar(name + "/num_zero/", var_num_zero)



#def f_summary_operation(var):
#    nodename=var.name.replace(":","_")
#    f_summary_one(var,nodename)

def f_summary_variables():
    varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in varlist:
        nodename=var.name.replace(":","_")
        f_summary_one(var,nodename)
def f_summary_gradients(loss,inlist=None):
    if inlist is None:
        varlist=tf.get_collection(default_summary_collection_gradient)
    else:
        varlist=inlist
    with tf.variable_scope("gradient_summary"):
        varlist2=zip(tf.gradients(loss,varlist),varlist)
        for var in varlist2:
            nodename=var[1].name.replace(":","_")+"_gradient"
            f_summary_one(var[0],nodename)
def f_summary_collection(inlist=None):
    namedict=dict()
    if inlist is None:
        varlist=tf.get_collection(default_summary_collection)
    else:
        varlist=inlist
    for var in varlist:
        nodename=var.name.replace(":","_")
        if namedict.has_key(nodename):
            namedict[nodename]=namedict[nodename]+1
            nodename=nodename+"_"+str(namedict[nodename])
        else:
            namedict[nodename]=1
        f_summary_one(var,nodename)

def f_summary_add_to_collection(var,collection=default_summary_collection):
    if isinstance(var,list):
        for x in var:
            tf.add_to_collection(collection, x)
    else:
        tf.add_to_collection(collection,var)

def f_summary_add_to_collection_gradient(var,collection=default_summary_collection_gradient):
    if isinstance(var,list):
        for x in var:
            tf.add_to_collection(collection, x)
    else:
        tf.add_to_collection(collection,var)
