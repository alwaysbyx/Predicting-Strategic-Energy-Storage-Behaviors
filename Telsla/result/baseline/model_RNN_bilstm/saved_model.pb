οχ5
΅
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
φ
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleιθelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleιθelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Όω3
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dP*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:dP*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:P*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:PP*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:P*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
·
-bidirectional/forward_lstm/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Θ*>
shared_name/-bidirectional/forward_lstm/lstm_cell_1/kernel
°
Abidirectional/forward_lstm/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOp-bidirectional/forward_lstm/lstm_cell_1/kernel*
_output_shapes
:	Θ*
dtype0
Λ
7bidirectional/forward_lstm/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Θ*H
shared_name97bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel
Δ
Kbidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp7bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel*
_output_shapes
:	2Θ*
dtype0
―
+bidirectional/forward_lstm/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Θ*<
shared_name-+bidirectional/forward_lstm/lstm_cell_1/bias
¨
?bidirectional/forward_lstm/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOp+bidirectional/forward_lstm/lstm_cell_1/bias*
_output_shapes	
:Θ*
dtype0
Ή
.bidirectional/backward_lstm/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Θ*?
shared_name0.bidirectional/backward_lstm/lstm_cell_2/kernel
²
Bbidirectional/backward_lstm/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOp.bidirectional/backward_lstm/lstm_cell_2/kernel*
_output_shapes
:	Θ*
dtype0
Ν
8bidirectional/backward_lstm/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Θ*I
shared_name:8bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel
Ζ
Lbidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp8bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel*
_output_shapes
:	2Θ*
dtype0
±
,bidirectional/backward_lstm/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Θ*=
shared_name.,bidirectional/backward_lstm/lstm_cell_2/bias
ͺ
@bidirectional/backward_lstm/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOp,bidirectional/backward_lstm/lstm_cell_2/bias*
_output_shapes	
:Θ*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dP*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:dP*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:P*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:PP*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:P*
dtype0
Ε
4Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Θ*E
shared_name64Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/m
Ύ
HAdam/bidirectional/forward_lstm/lstm_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp4Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/m*
_output_shapes
:	Θ*
dtype0
Ω
>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Θ*O
shared_name@>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/m
?
RAdam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/m*
_output_shapes
:	2Θ*
dtype0
½
2Adam/bidirectional/forward_lstm/lstm_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Θ*C
shared_name42Adam/bidirectional/forward_lstm/lstm_cell_1/bias/m
Ά
FAdam/bidirectional/forward_lstm/lstm_cell_1/bias/m/Read/ReadVariableOpReadVariableOp2Adam/bidirectional/forward_lstm/lstm_cell_1/bias/m*
_output_shapes	
:Θ*
dtype0
Η
5Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Θ*F
shared_name75Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/m
ΐ
IAdam/bidirectional/backward_lstm/lstm_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/m*
_output_shapes
:	Θ*
dtype0
Ϋ
?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Θ*P
shared_nameA?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/m
Τ
SAdam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/m*
_output_shapes
:	2Θ*
dtype0
Ώ
3Adam/bidirectional/backward_lstm/lstm_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Θ*D
shared_name53Adam/bidirectional/backward_lstm/lstm_cell_2/bias/m
Έ
GAdam/bidirectional/backward_lstm/lstm_cell_2/bias/m/Read/ReadVariableOpReadVariableOp3Adam/bidirectional/backward_lstm/lstm_cell_2/bias/m*
_output_shapes	
:Θ*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dP*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:dP*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:P*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:PP*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:P*
dtype0
Ε
4Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Θ*E
shared_name64Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/v
Ύ
HAdam/bidirectional/forward_lstm/lstm_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp4Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/v*
_output_shapes
:	Θ*
dtype0
Ω
>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Θ*O
shared_name@>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/v
?
RAdam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/v*
_output_shapes
:	2Θ*
dtype0
½
2Adam/bidirectional/forward_lstm/lstm_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Θ*C
shared_name42Adam/bidirectional/forward_lstm/lstm_cell_1/bias/v
Ά
FAdam/bidirectional/forward_lstm/lstm_cell_1/bias/v/Read/ReadVariableOpReadVariableOp2Adam/bidirectional/forward_lstm/lstm_cell_1/bias/v*
_output_shapes	
:Θ*
dtype0
Η
5Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Θ*F
shared_name75Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/v
ΐ
IAdam/bidirectional/backward_lstm/lstm_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/v*
_output_shapes
:	Θ*
dtype0
Ϋ
?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Θ*P
shared_nameA?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/v
Τ
SAdam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/v*
_output_shapes
:	2Θ*
dtype0
Ώ
3Adam/bidirectional/backward_lstm/lstm_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Θ*D
shared_name53Adam/bidirectional/backward_lstm/lstm_cell_2/bias/v
Έ
GAdam/bidirectional/backward_lstm/lstm_cell_2/bias/v/Read/ReadVariableOpReadVariableOp3Adam/bidirectional/backward_lstm/lstm_cell_2/bias/v*
_output_shapes	
:Θ*
dtype0

NoOpNoOp
U
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ΐT
valueΆTB³T B¬T
Ϋ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
·
forward_layer
backward_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses* 
₯
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)_random_generator
*__call__
*+&call_and_return_all_conditional_losses* 
¦

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*

4iter

5beta_1

6beta_2
	7decay
8learning_ratemm,m-m9m:m ;m‘<m’=m£>m€v₯v¦,v§-v¨9v©:vͺ;v«<v¬=v­>v?*
J
90
:1
;2
<3
=4
>5
6
7
,8
-9*
J
90
:1
;2
<3
=4
>5
6
7
,8
-9*
* 
°
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Dserving_default* 
Α
Ecell
F
state_spec
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K_random_generator
L__call__
*M&call_and_return_all_conditional_losses*
Α
Ncell
O
state_spec
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T_random_generator
U__call__
*V&call_and_return_all_conditional_losses*
.
90
:1
;2
<3
=4
>5*
.
90
:1
;2
<3
=4
>5*
* 

Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
%	variables
&trainable_variables
'regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-bidirectional/forward_lstm/lstm_cell_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE7bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+bidirectional/forward_lstm/lstm_cell_1/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.bidirectional/backward_lstm/lstm_cell_2/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,bidirectional/backward_lstm/lstm_cell_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

p0*
* 
* 
* 
γ
q
state_size

9kernel
:recurrent_kernel
;bias
r	variables
strainable_variables
tregularization_losses
u	keras_api
v_random_generator
w__call__
*x&call_and_return_all_conditional_losses*
* 

90
:1
;2*

90
:1
;2*
* 


ystates
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
* 
* 
* 
κ

state_size

<kernel
=recurrent_kernel
>bias
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses*
* 

<0
=1
>2*

<0
=1
>2*
* 
₯
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
* 

90
:1
;2*

90
:1
;2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
r	variables
strainable_variables
tregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

E0*
* 
* 
* 
* 

<0
=1
>2*

<0
=1
>2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

N0*
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/bidirectional/forward_lstm/lstm_cell_1/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE5Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/bidirectional/backward_lstm/lstm_cell_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/bidirectional/forward_lstm/lstm_cell_1/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE5Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/bidirectional/backward_lstm/lstm_cell_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

#serving_default_bidirectional_inputPlaceholder*+
_output_shapes
:?????????P*
dtype0* 
shape:?????????P
Ά
StatefulPartitionedCallStatefulPartitionedCall#serving_default_bidirectional_input-bidirectional/forward_lstm/lstm_cell_1/kernel7bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel+bidirectional/forward_lstm/lstm_cell_1/bias.bidirectional/backward_lstm/lstm_cell_2/kernel8bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel,bidirectional/backward_lstm/lstm_cell_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_41103
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
±
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpAbidirectional/forward_lstm/lstm_cell_1/kernel/Read/ReadVariableOpKbidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp?bidirectional/forward_lstm/lstm_cell_1/bias/Read/ReadVariableOpBbidirectional/backward_lstm/lstm_cell_2/kernel/Read/ReadVariableOpLbidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/Read/ReadVariableOp@bidirectional/backward_lstm/lstm_cell_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOpHAdam/bidirectional/forward_lstm/lstm_cell_1/kernel/m/Read/ReadVariableOpRAdam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOpFAdam/bidirectional/forward_lstm/lstm_cell_1/bias/m/Read/ReadVariableOpIAdam/bidirectional/backward_lstm/lstm_cell_2/kernel/m/Read/ReadVariableOpSAdam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOpGAdam/bidirectional/backward_lstm/lstm_cell_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpHAdam/bidirectional/forward_lstm/lstm_cell_1/kernel/v/Read/ReadVariableOpRAdam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOpFAdam/bidirectional/forward_lstm/lstm_cell_1/bias/v/Read/ReadVariableOpIAdam/bidirectional/backward_lstm/lstm_cell_2/kernel/v/Read/ReadVariableOpSAdam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOpGAdam/bidirectional/backward_lstm/lstm_cell_2/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_43960
Θ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate-bidirectional/forward_lstm/lstm_cell_1/kernel7bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel+bidirectional/forward_lstm/lstm_cell_1/bias.bidirectional/backward_lstm/lstm_cell_2/kernel8bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel,bidirectional/backward_lstm/lstm_cell_2/biastotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m4Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/m>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/m2Adam/bidirectional/forward_lstm/lstm_cell_1/bias/m5Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/m?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/m3Adam/bidirectional/backward_lstm/lstm_cell_2/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v4Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/v>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/v2Adam/bidirectional/forward_lstm/lstm_cell_1/bias/v5Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/v?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/v3Adam/bidirectional/backward_lstm/lstm_cell_2/bias/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_44081Μ€2
°
Ύ
while_cond_43545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_43545___redundant_placeholder03
/while_while_cond_43545___redundant_placeholder13
/while_while_cond_43545___redundant_placeholder23
/while_while_cond_43545___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ε	
σ
B__inference_dense_1_layer_call_and_return_conditional_losses_42390

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
Οe
ͺ
7sequential_bidirectional_backward_lstm_while_body_38006j
fsequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_loop_counterp
lsequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_maximum_iterations<
8sequential_bidirectional_backward_lstm_while_placeholder>
:sequential_bidirectional_backward_lstm_while_placeholder_1>
:sequential_bidirectional_backward_lstm_while_placeholder_2>
:sequential_bidirectional_backward_lstm_while_placeholder_3i
esequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_strided_slice_1_0¦
‘sequential_bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0l
Ysequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	Θn
[sequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2Θi
Zsequential_bidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ9
5sequential_bidirectional_backward_lstm_while_identity;
7sequential_bidirectional_backward_lstm_while_identity_1;
7sequential_bidirectional_backward_lstm_while_identity_2;
7sequential_bidirectional_backward_lstm_while_identity_3;
7sequential_bidirectional_backward_lstm_while_identity_4;
7sequential_bidirectional_backward_lstm_while_identity_5g
csequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_strided_slice_1€
sequential_bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensorj
Wsequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	Θl
Ysequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θg
Xsequential_bidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’Osequential/bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’Nsequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’Psequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp―
^sequential/bidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   κ
Psequential/bidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem‘sequential_bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_08sequential_bidirectional_backward_lstm_while_placeholdergsequential/bidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ι
Nsequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpYsequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0­
?sequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMulMatMulWsequential/bidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Vsequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θν
Psequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp[sequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
Asequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1MatMul:sequential_bidirectional_backward_lstm_while_placeholder_2Xsequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
<sequential/bidirectional/backward_lstm/while/lstm_cell_2/addAddV2Isequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul:product:0Ksequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θη
Osequential/bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpZsequential_bidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0
@sequential/bidirectional/backward_lstm/while/lstm_cell_2/BiasAddBiasAdd@sequential/bidirectional/backward_lstm/while/lstm_cell_2/add:z:0Wsequential/bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
Hsequential/bidirectional/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :α
>sequential/bidirectional/backward_lstm/while/lstm_cell_2/splitSplitQsequential/bidirectional/backward_lstm/while/lstm_cell_2/split/split_dim:output:0Isequential/bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitΖ
@sequential/bidirectional/backward_lstm/while/lstm_cell_2/SigmoidSigmoidGsequential/bidirectional/backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2Θ
Bsequential/bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_1SigmoidGsequential/bidirectional/backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2ω
<sequential/bidirectional/backward_lstm/while/lstm_cell_2/mulMulFsequential/bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0:sequential_bidirectional_backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2ΐ
=sequential/bidirectional/backward_lstm/while/lstm_cell_2/ReluReluGsequential/bidirectional/backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
>sequential/bidirectional/backward_lstm/while/lstm_cell_2/mul_1MulDsequential/bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid:y:0Ksequential/bidirectional/backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2?
>sequential/bidirectional/backward_lstm/while/lstm_cell_2/add_1AddV2@sequential/bidirectional/backward_lstm/while/lstm_cell_2/mul:z:0Bsequential/bidirectional/backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2Θ
Bsequential/bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_2SigmoidGsequential/bidirectional/backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2½
?sequential/bidirectional/backward_lstm/while/lstm_cell_2/Relu_1ReluBsequential/bidirectional/backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
>sequential/bidirectional/backward_lstm/while/lstm_cell_2/mul_2MulFsequential/bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_2:y:0Msequential/bidirectional/backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ΰ
Qsequential/bidirectional/backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem:sequential_bidirectional_backward_lstm_while_placeholder_18sequential_bidirectional_backward_lstm_while_placeholderBsequential/bidirectional/backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?t
2sequential/bidirectional/backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ρ
0sequential/bidirectional/backward_lstm/while/addAddV28sequential_bidirectional_backward_lstm_while_placeholder;sequential/bidirectional/backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: v
4sequential/bidirectional/backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
2sequential/bidirectional/backward_lstm/while/add_1AddV2fsequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_loop_counter=sequential/bidirectional/backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: Ξ
5sequential/bidirectional/backward_lstm/while/IdentityIdentity6sequential/bidirectional/backward_lstm/while/add_1:z:02^sequential/bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: 
7sequential/bidirectional/backward_lstm/while/Identity_1Identitylsequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_maximum_iterations2^sequential/bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: Ξ
7sequential/bidirectional/backward_lstm/while/Identity_2Identity4sequential/bidirectional/backward_lstm/while/add:z:02^sequential/bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: 
7sequential/bidirectional/backward_lstm/while/Identity_3Identityasequential/bidirectional/backward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^sequential/bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?ν
7sequential/bidirectional/backward_lstm/while/Identity_4IdentityBsequential/bidirectional/backward_lstm/while/lstm_cell_2/mul_2:z:02^sequential/bidirectional/backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2ν
7sequential/bidirectional/backward_lstm/while/Identity_5IdentityBsequential/bidirectional/backward_lstm/while/lstm_cell_2/add_1:z:02^sequential/bidirectional/backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2ι
1sequential/bidirectional/backward_lstm/while/NoOpNoOpP^sequential/bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpO^sequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpQ^sequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "w
5sequential_bidirectional_backward_lstm_while_identity>sequential/bidirectional/backward_lstm/while/Identity:output:0"{
7sequential_bidirectional_backward_lstm_while_identity_1@sequential/bidirectional/backward_lstm/while/Identity_1:output:0"{
7sequential_bidirectional_backward_lstm_while_identity_2@sequential/bidirectional/backward_lstm/while/Identity_2:output:0"{
7sequential_bidirectional_backward_lstm_while_identity_3@sequential/bidirectional/backward_lstm/while/Identity_3:output:0"{
7sequential_bidirectional_backward_lstm_while_identity_4@sequential/bidirectional/backward_lstm/while/Identity_4:output:0"{
7sequential_bidirectional_backward_lstm_while_identity_5@sequential/bidirectional/backward_lstm/while/Identity_5:output:0"Ά
Xsequential_bidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceZsequential_bidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"Έ
Ysequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource[sequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"΄
Wsequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resourceYsequential_bidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"Μ
csequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_strided_slice_1esequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_strided_slice_1_0"Ζ
sequential_bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor‘sequential_bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2’
Osequential/bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpOsequential/bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2 
Nsequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpNsequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2€
Psequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpPsequential/bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ή
Φ
backward_lstm_while_cond_413708
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_3:
6backward_lstm_while_less_backward_lstm_strided_slice_1O
Kbackward_lstm_while_backward_lstm_while_cond_41370___redundant_placeholder0O
Kbackward_lstm_while_backward_lstm_while_cond_41370___redundant_placeholder1O
Kbackward_lstm_while_backward_lstm_while_cond_41370___redundant_placeholder2O
Kbackward_lstm_while_backward_lstm_while_cond_41370___redundant_placeholder3 
backward_lstm_while_identity

backward_lstm/while/LessLessbackward_lstm_while_placeholder6backward_lstm_while_less_backward_lstm_strided_slice_1*
T0*
_output_shapes
: g
backward_lstm/while/IdentityIdentitybackward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
°
Ύ
while_cond_43110
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_43110___redundant_placeholder03
/while_while_cond_43110___redundant_placeholder13
/while_while_cond_43110___redundant_placeholder23
/while_while_cond_43110___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Τ
Π
E__inference_sequential_layer_call_and_return_conditional_losses_40307

inputs&
bidirectional_40281:	Θ&
bidirectional_40283:	2Θ"
bidirectional_40285:	Θ&
bidirectional_40287:	Θ&
bidirectional_40289:	2Θ"
bidirectional_40291:	Θ
dense_40294:dP
dense_40296:P
dense_1_40301:PP
dense_1_40303:P
identity’%bidirectional/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dropout/StatefulPartitionedCallέ
%bidirectional/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_40281bidirectional_40283bidirectional_40285bidirectional_40287bidirectional_40289bidirectional_40291*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_40236
dense/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_40294dense_40296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_39822Ϋ
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_39833β
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_39912
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_40301dense_1_40303*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_39852w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp&^bidirectional/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs

Β
forward_lstm_while_cond_395706
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_38
4forward_lstm_while_less_forward_lstm_strided_slice_1M
Iforward_lstm_while_forward_lstm_while_cond_39570___redundant_placeholder0M
Iforward_lstm_while_forward_lstm_while_cond_39570___redundant_placeholder1M
Iforward_lstm_while_forward_lstm_while_cond_39570___redundant_placeholder2M
Iforward_lstm_while_forward_lstm_while_cond_39570___redundant_placeholder3
forward_lstm_while_identity

forward_lstm/while/LessLessforward_lstm_while_placeholder4forward_lstm_while_less_forward_lstm_strided_slice_1*
T0*
_output_shapes
: e
forward_lstm/while/IdentityIdentityforward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ή
Φ
backward_lstm_while_cond_422288
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_3:
6backward_lstm_while_less_backward_lstm_strided_slice_1O
Kbackward_lstm_while_backward_lstm_while_cond_42228___redundant_placeholder0O
Kbackward_lstm_while_backward_lstm_while_cond_42228___redundant_placeholder1O
Kbackward_lstm_while_backward_lstm_while_cond_42228___redundant_placeholder2O
Kbackward_lstm_while_backward_lstm_while_cond_42228___redundant_placeholder3 
backward_lstm_while_identity

backward_lstm/while/LessLessbackward_lstm_while_placeholder6backward_lstm_while_less_backward_lstm_strided_slice_1*
T0*
_output_shapes
: g
backward_lstm/while/IdentityIdentitybackward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
σ

H__inference_bidirectional_layer_call_and_return_conditional_losses_39122

inputs%
forward_lstm_38960:	Θ%
forward_lstm_38962:	2Θ!
forward_lstm_38964:	Θ&
backward_lstm_39112:	Θ&
backward_lstm_39114:	2Θ"
backward_lstm_39116:	Θ
identity’%backward_lstm/StatefulPartitionedCall’$forward_lstm/StatefulPartitionedCall
$forward_lstm/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_38960forward_lstm_38962forward_lstm_38964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_forward_lstm_layer_call_and_return_conditional_losses_38959
%backward_lstm/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_39112backward_lstm_39114backward_lstm_39116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_39111M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Β
concatConcatV2-forward_lstm/StatefulPartitionedCall:output:0.backward_lstm/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????d
NoOpNoOp&^backward_lstm/StatefulPartitionedCall%^forward_lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2N
%backward_lstm/StatefulPartitionedCall%backward_lstm/StatefulPartitionedCall2L
$forward_lstm/StatefulPartitionedCall$forward_lstm/StatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
«U
τ
__inference__traced_save_43960
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopL
Hsavev2_bidirectional_forward_lstm_lstm_cell_1_kernel_read_readvariableopV
Rsavev2_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_read_readvariableopJ
Fsavev2_bidirectional_forward_lstm_lstm_cell_1_bias_read_readvariableopM
Isavev2_bidirectional_backward_lstm_lstm_cell_2_kernel_read_readvariableopW
Ssavev2_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_read_readvariableopK
Gsavev2_bidirectional_backward_lstm_lstm_cell_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopS
Osavev2_adam_bidirectional_forward_lstm_lstm_cell_1_kernel_m_read_readvariableop]
Ysavev2_adam_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_m_read_readvariableopQ
Msavev2_adam_bidirectional_forward_lstm_lstm_cell_1_bias_m_read_readvariableopT
Psavev2_adam_bidirectional_backward_lstm_lstm_cell_2_kernel_m_read_readvariableop^
Zsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_m_read_readvariableopR
Nsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopS
Osavev2_adam_bidirectional_forward_lstm_lstm_cell_1_kernel_v_read_readvariableop]
Ysavev2_adam_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_v_read_readvariableopQ
Msavev2_adam_bidirectional_forward_lstm_lstm_cell_1_bias_v_read_readvariableopT
Psavev2_adam_bidirectional_backward_lstm_lstm_cell_2_kernel_v_read_readvariableop^
Zsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_v_read_readvariableopR
Nsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ο
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*
valueB&B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHΉ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Κ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopHsavev2_bidirectional_forward_lstm_lstm_cell_1_kernel_read_readvariableopRsavev2_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_read_readvariableopFsavev2_bidirectional_forward_lstm_lstm_cell_1_bias_read_readvariableopIsavev2_bidirectional_backward_lstm_lstm_cell_2_kernel_read_readvariableopSsavev2_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_read_readvariableopGsavev2_bidirectional_backward_lstm_lstm_cell_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopOsavev2_adam_bidirectional_forward_lstm_lstm_cell_1_kernel_m_read_readvariableopYsavev2_adam_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_m_read_readvariableopMsavev2_adam_bidirectional_forward_lstm_lstm_cell_1_bias_m_read_readvariableopPsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_kernel_m_read_readvariableopZsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_m_read_readvariableopNsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopOsavev2_adam_bidirectional_forward_lstm_lstm_cell_1_kernel_v_read_readvariableopYsavev2_adam_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_v_read_readvariableopMsavev2_adam_bidirectional_forward_lstm_lstm_cell_1_bias_v_read_readvariableopPsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_kernel_v_read_readvariableopZsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_v_read_readvariableopNsavev2_adam_bidirectional_backward_lstm_lstm_cell_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*΅
_input_shapes£
 : :dP:P:PP:P: : : : : :	Θ:	2Θ:Θ:	Θ:	2Θ:Θ: : :dP:P:PP:P:	Θ:	2Θ:Θ:	Θ:	2Θ:Θ:dP:P:PP:P:	Θ:	2Θ:Θ:	Θ:	2Θ:Θ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:dP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	Θ:%!

_output_shapes
:	2Θ:!

_output_shapes	
:Θ:%!

_output_shapes
:	Θ:%!

_output_shapes
:	2Θ:!

_output_shapes	
:Θ:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:dP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:%!

_output_shapes
:	Θ:%!

_output_shapes
:	2Θ:!

_output_shapes	
:Θ:%!

_output_shapes
:	Θ:%!

_output_shapes
:	2Θ:!

_output_shapes	
:Θ:$ 

_output_shapes

:dP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:% !

_output_shapes
:	Θ:%!!

_output_shapes
:	2Θ:!"

_output_shapes	
:Θ:%#!

_output_shapes
:	Θ:%$!

_output_shapes
:	2Θ:!%

_output_shapes	
:Θ:&

_output_shapes
: 
Φ9

H__inference_backward_lstm_layer_call_and_return_conditional_losses_38608

inputs$
lstm_cell_2_38526:	Θ$
lstm_cell_2_38528:	2Θ 
lstm_cell_2_38530:	Θ
identity’#lstm_cell_2/StatefulPartitionedCall’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ε
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskμ
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_38526lstm_cell_2_38528lstm_cell_2_38530*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_38525n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ―
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_38526lstm_cell_2_38528lstm_cell_2_38530*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38539*
condR
while_cond_38538*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2t
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
G
ζ
forward_lstm_while_body_420886
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_35
1forward_lstm_while_forward_lstm_strided_slice_1_0q
mforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0R
?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘT
Aforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘO
@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
forward_lstm_while_identity!
forward_lstm_while_identity_1!
forward_lstm_while_identity_2!
forward_lstm_while_identity_3!
forward_lstm_while_identity_4!
forward_lstm_while_identity_53
/forward_lstm_while_forward_lstm_strided_slice_1o
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensorP
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	ΘR
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘM
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp
Dforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   η
6forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0forward_lstm_while_placeholderMforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0΅
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0ί
%forward_lstm/while/lstm_cell_1/MatMulMatMul=forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΉ
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ζ
'forward_lstm/while/lstm_cell_1/MatMul_1MatMul forward_lstm_while_placeholder_2>forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΒ
"forward_lstm/while/lstm_cell_1/addAddV2/forward_lstm/while/lstm_cell_1/MatMul:product:01forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ³
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Λ
&forward_lstm/while/lstm_cell_1/BiasAddBiasAdd&forward_lstm/while/lstm_cell_1/add:z:0=forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θp
.forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$forward_lstm/while/lstm_cell_1/splitSplit7forward_lstm/while/lstm_cell_1/split/split_dim:output:0/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
&forward_lstm/while/lstm_cell_1/SigmoidSigmoid-forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_1Sigmoid-forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2«
"forward_lstm/while/lstm_cell_1/mulMul,forward_lstm/while/lstm_cell_1/Sigmoid_1:y:0 forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
#forward_lstm/while/lstm_cell_1/ReluRelu-forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2Ό
$forward_lstm/while/lstm_cell_1/mul_1Mul*forward_lstm/while/lstm_cell_1/Sigmoid:y:01forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2±
$forward_lstm/while/lstm_cell_1/add_1AddV2&forward_lstm/while/lstm_cell_1/mul:z:0(forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_2Sigmoid-forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
%forward_lstm/while/lstm_cell_1/Relu_1Relu(forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2ΐ
$forward_lstm/while/lstm_cell_1/mul_2Mul,forward_lstm/while/lstm_cell_1/Sigmoid_2:y:03forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ψ
7forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem forward_lstm_while_placeholder_1forward_lstm_while_placeholder(forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?Z
forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/addAddV2forward_lstm_while_placeholder!forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: \
forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/add_1AddV22forward_lstm_while_forward_lstm_while_loop_counter#forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_lstm/while/IdentityIdentityforward_lstm/while/add_1:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_1Identity8forward_lstm_while_forward_lstm_while_maximum_iterations^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_2Identityforward_lstm/while/add:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: ΐ
forward_lstm/while/Identity_3IdentityGforward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?
forward_lstm/while/Identity_4Identity(forward_lstm/while/lstm_cell_1/mul_2:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/Identity_5Identity(forward_lstm/while/lstm_cell_1/add_1:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/NoOpNoOp6^forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5^forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp7^forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/forward_lstm_while_forward_lstm_strided_slice_11forward_lstm_while_forward_lstm_strided_slice_1_0"C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0"G
forward_lstm_while_identity_1&forward_lstm/while/Identity_1:output:0"G
forward_lstm_while_identity_2&forward_lstm/while/Identity_2:output:0"G
forward_lstm_while_identity_3&forward_lstm/while/Identity_3:output:0"G
forward_lstm_while_identity_4&forward_lstm/while/Identity_4:output:0"G
forward_lstm_while_identity_5&forward_lstm/while/Identity_5:output:0"
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0"
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"ά
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensormforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2n
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2l
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2p
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ή
Φ
backward_lstm_while_cond_419428
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_3:
6backward_lstm_while_less_backward_lstm_strided_slice_1O
Kbackward_lstm_while_backward_lstm_while_cond_41942___redundant_placeholder0O
Kbackward_lstm_while_backward_lstm_while_cond_41942___redundant_placeholder1O
Kbackward_lstm_while_backward_lstm_while_cond_41942___redundant_placeholder2O
Kbackward_lstm_while_backward_lstm_while_cond_41942___redundant_placeholder3 
backward_lstm_while_identity

backward_lstm/while/LessLessbackward_lstm_while_placeholder6backward_lstm_while_less_backward_lstm_strided_slice_1*
T0*
_output_shapes
: g
backward_lstm/while/IdentityIdentitybackward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:

Β
forward_lstm_while_cond_400086
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_38
4forward_lstm_while_less_forward_lstm_strided_slice_1M
Iforward_lstm_while_forward_lstm_while_cond_40008___redundant_placeholder0M
Iforward_lstm_while_forward_lstm_while_cond_40008___redundant_placeholder1M
Iforward_lstm_while_forward_lstm_while_cond_40008___redundant_placeholder2M
Iforward_lstm_while_forward_lstm_while_cond_40008___redundant_placeholder3
forward_lstm_while_identity

forward_lstm/while/LessLessforward_lstm_while_placeholder4forward_lstm_while_less_forward_lstm_strided_slice_1*
T0*
_output_shapes
: e
forward_lstm/while/IdentityIdentityforward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
°
Ύ
while_cond_38874
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38874___redundant_placeholder03
/while_while_cond_38874___redundant_placeholder13
/while_while_cond_38874___redundant_placeholder23
/while_while_cond_38874___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ν


*__inference_sequential_layer_call_fn_40355
bidirectional_input
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
	unknown_5:dP
	unknown_6:P
	unknown_7:PP
	unknown_8:P
identity’StatefulPartitionedCallΟ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_40307o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????P
-
_user_specified_namebidirectional_input
Π"
Υ
while_body_38187
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_1_38211_0:	Θ,
while_lstm_cell_1_38213_0:	2Θ(
while_lstm_cell_1_38215_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_1_38211:	Θ*
while_lstm_cell_1_38213:	2Θ&
while_lstm_cell_1_38215:	Θ’)while/lstm_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ͺ
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_38211_0while_lstm_cell_1_38213_0while_lstm_cell_1_38215_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_38173Ϋ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????2x

while/NoOpNoOp*^while/lstm_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_1_38211while_lstm_cell_1_38211_0"4
while_lstm_cell_1_38213while_lstm_cell_1_38213_0"4
while_lstm_cell_1_38215while_lstm_cell_1_38215_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Σ

F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_38525

inputs

states
states_11
matmul_readvariableop_resource:	Θ3
 matmul_1_readvariableop_resource:	2Θ.
biasadd_readvariableop_resource:	Θ
identity

identity_1

identity_2’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ά
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
H

backward_lstm_while_body_422298
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_37
3backward_lstm_while_backward_lstm_strided_slice_1_0s
obackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0S
@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘU
Bbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘP
Abackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ 
backward_lstm_while_identity"
backward_lstm_while_identity_1"
backward_lstm_while_identity_2"
backward_lstm_while_identity_3"
backward_lstm_while_identity_4"
backward_lstm_while_identity_55
1backward_lstm_while_backward_lstm_strided_slice_1q
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorQ
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	ΘS
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘN
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp
Ebackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   μ
7backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0backward_lstm_while_placeholderNbackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0·
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0β
&backward_lstm/while/lstm_cell_2/MatMulMatMul>backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ»
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ι
(backward_lstm/while/lstm_cell_2/MatMul_1MatMul!backward_lstm_while_placeholder_2?backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΕ
#backward_lstm/while/lstm_cell_2/addAddV20backward_lstm/while/lstm_cell_2/MatMul:product:02backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ΅
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Ξ
'backward_lstm/while/lstm_cell_2/BiasAddBiasAdd'backward_lstm/while/lstm_cell_2/add:z:0>backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θq
/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%backward_lstm/while/lstm_cell_2/splitSplit8backward_lstm/while/lstm_cell_2/split/split_dim:output:00backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
'backward_lstm/while/lstm_cell_2/SigmoidSigmoid.backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_1Sigmoid.backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2?
#backward_lstm/while/lstm_cell_2/mulMul-backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0!backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
$backward_lstm/while/lstm_cell_2/ReluRelu.backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2Ώ
%backward_lstm/while/lstm_cell_2/mul_1Mul+backward_lstm/while/lstm_cell_2/Sigmoid:y:02backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2΄
%backward_lstm/while/lstm_cell_2/add_1AddV2'backward_lstm/while/lstm_cell_2/mul:z:0)backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_2Sigmoid.backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
&backward_lstm/while/lstm_cell_2/Relu_1Relu)backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2Γ
%backward_lstm/while/lstm_cell_2/mul_2Mul-backward_lstm/while/lstm_cell_2/Sigmoid_2:y:04backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ό
8backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!backward_lstm_while_placeholder_1backward_lstm_while_placeholder)backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?[
backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/addAddV2backward_lstm_while_placeholder"backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: ]
backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/add_1AddV24backward_lstm_while_backward_lstm_while_loop_counter$backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_lstm/while/IdentityIdentitybackward_lstm/while/add_1:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: ’
backward_lstm/while/Identity_1Identity:backward_lstm_while_backward_lstm_while_maximum_iterations^backward_lstm/while/NoOp*
T0*
_output_shapes
: 
backward_lstm/while/Identity_2Identitybackward_lstm/while/add:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: Γ
backward_lstm/while/Identity_3IdentityHbackward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?’
backward_lstm/while/Identity_4Identity)backward_lstm/while/lstm_cell_2/mul_2:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2’
backward_lstm/while/Identity_5Identity)backward_lstm/while/lstm_cell_2/add_1:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
backward_lstm/while/NoOpNoOp7^backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6^backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp8^backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1backward_lstm_while_backward_lstm_strided_slice_13backward_lstm_while_backward_lstm_strided_slice_1_0"E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0"I
backward_lstm_while_identity_1'backward_lstm/while/Identity_1:output:0"I
backward_lstm_while_identity_2'backward_lstm/while/Identity_2:output:0"I
backward_lstm_while_identity_3'backward_lstm/while/Identity_3:output:0"I
backward_lstm_while_identity_4'backward_lstm/while/Identity_4:output:0"I
backward_lstm_while_identity_5'backward_lstm/while/Identity_5:output:0"
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resourceBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"ΰ
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2p
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2n
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2r
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 

Β
forward_lstm_while_cond_412296
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_38
4forward_lstm_while_less_forward_lstm_strided_slice_1M
Iforward_lstm_while_forward_lstm_while_cond_41229___redundant_placeholder0M
Iforward_lstm_while_forward_lstm_while_cond_41229___redundant_placeholder1M
Iforward_lstm_while_forward_lstm_while_cond_41229___redundant_placeholder2M
Iforward_lstm_while_forward_lstm_while_cond_41229___redundant_placeholder3
forward_lstm_while_identity

forward_lstm/while/LessLessforward_lstm_while_placeholder4forward_lstm_while_less_forward_lstm_strided_slice_1*
T0*
_output_shapes
: e
forward_lstm/while/IdentityIdentityforward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
‘

?
#__inference_signature_wrapper_41103
bidirectional_input
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
	unknown_5:dP
	unknown_6:P
	unknown_7:PP
	unknown_8:P
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_38106o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????P
-
_user_specified_namebidirectional_input
άX
Κ
,bidirectional_backward_lstm_while_body_40969T
Pbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_loop_counterZ
Vbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_maximum_iterations1
-bidirectional_backward_lstm_while_placeholder3
/bidirectional_backward_lstm_while_placeholder_13
/bidirectional_backward_lstm_while_placeholder_23
/bidirectional_backward_lstm_while_placeholder_3S
Obidirectional_backward_lstm_while_bidirectional_backward_lstm_strided_slice_1_0
bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0a
Nbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	Θc
Pbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2Θ^
Obidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ.
*bidirectional_backward_lstm_while_identity0
,bidirectional_backward_lstm_while_identity_10
,bidirectional_backward_lstm_while_identity_20
,bidirectional_backward_lstm_while_identity_30
,bidirectional_backward_lstm_while_identity_40
,bidirectional_backward_lstm_while_identity_5Q
Mbidirectional_backward_lstm_while_bidirectional_backward_lstm_strided_slice_1
bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_
Lbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	Θa
Nbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θ\
Mbidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’Dbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’Cbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’Ebidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp€
Sbidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ³
Ebidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0-bidirectional_backward_lstm_while_placeholder\bidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0Σ
Cbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpNbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0
4bidirectional/backward_lstm/while/lstm_cell_2/MatMulMatMulLbidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Kbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΧ
Ebidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpPbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0σ
6bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1MatMul/bidirectional_backward_lstm_while_placeholder_2Mbidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θο
1bidirectional/backward_lstm/while/lstm_cell_2/addAddV2>bidirectional/backward_lstm/while/lstm_cell_2/MatMul:product:0@bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΡ
Dbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpObidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0ψ
5bidirectional/backward_lstm/while/lstm_cell_2/BiasAddBiasAdd5bidirectional/backward_lstm/while/lstm_cell_2/add:z:0Lbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
=bidirectional/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ΐ
3bidirectional/backward_lstm/while/lstm_cell_2/splitSplitFbidirectional/backward_lstm/while/lstm_cell_2/split/split_dim:output:0>bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split°
5bidirectional/backward_lstm/while/lstm_cell_2/SigmoidSigmoid<bidirectional/backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2²
7bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_1Sigmoid<bidirectional/backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2Ψ
1bidirectional/backward_lstm/while/lstm_cell_2/mulMul;bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0/bidirectional_backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2ͺ
2bidirectional/backward_lstm/while/lstm_cell_2/ReluRelu<bidirectional/backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2ι
3bidirectional/backward_lstm/while/lstm_cell_2/mul_1Mul9bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid:y:0@bidirectional/backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2ή
3bidirectional/backward_lstm/while/lstm_cell_2/add_1AddV25bidirectional/backward_lstm/while/lstm_cell_2/mul:z:07bidirectional/backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2²
7bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_2Sigmoid<bidirectional/backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2§
4bidirectional/backward_lstm/while/lstm_cell_2/Relu_1Relu7bidirectional/backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2ν
3bidirectional/backward_lstm/while/lstm_cell_2/mul_2Mul;bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_2:y:0Bbidirectional/backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2΄
Fbidirectional/backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/bidirectional_backward_lstm_while_placeholder_1-bidirectional_backward_lstm_while_placeholder7bidirectional/backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?i
'bidirectional/backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :°
%bidirectional/backward_lstm/while/addAddV2-bidirectional_backward_lstm_while_placeholder0bidirectional/backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: k
)bidirectional/backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Χ
'bidirectional/backward_lstm/while/add_1AddV2Pbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_loop_counter2bidirectional/backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: ­
*bidirectional/backward_lstm/while/IdentityIdentity+bidirectional/backward_lstm/while/add_1:z:0'^bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: Ϊ
,bidirectional/backward_lstm/while/Identity_1IdentityVbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_maximum_iterations'^bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: ­
,bidirectional/backward_lstm/while/Identity_2Identity)bidirectional/backward_lstm/while/add:z:0'^bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: ν
,bidirectional/backward_lstm/while/Identity_3IdentityVbidirectional/backward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?Μ
,bidirectional/backward_lstm/while/Identity_4Identity7bidirectional/backward_lstm/while/lstm_cell_2/mul_2:z:0'^bidirectional/backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2Μ
,bidirectional/backward_lstm/while/Identity_5Identity7bidirectional/backward_lstm/while/lstm_cell_2/add_1:z:0'^bidirectional/backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2½
&bidirectional/backward_lstm/while/NoOpNoOpE^bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpD^bidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpF^bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 " 
Mbidirectional_backward_lstm_while_bidirectional_backward_lstm_strided_slice_1Obidirectional_backward_lstm_while_bidirectional_backward_lstm_strided_slice_1_0"a
*bidirectional_backward_lstm_while_identity3bidirectional/backward_lstm/while/Identity:output:0"e
,bidirectional_backward_lstm_while_identity_15bidirectional/backward_lstm/while/Identity_1:output:0"e
,bidirectional_backward_lstm_while_identity_25bidirectional/backward_lstm/while/Identity_2:output:0"e
,bidirectional_backward_lstm_while_identity_35bidirectional/backward_lstm/while/Identity_3:output:0"e
,bidirectional_backward_lstm_while_identity_45bidirectional/backward_lstm/while/Identity_4:output:0"e
,bidirectional_backward_lstm_while_identity_55bidirectional/backward_lstm/while/Identity_5:output:0" 
Mbidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceObidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"’
Nbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resourcePbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
Lbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resourceNbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"
bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensorbidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2
Dbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpDbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2
Cbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpCbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2
Ebidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpEbidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
ΐJ

G__inference_forward_lstm_layer_call_and_return_conditional_losses_39460

inputs=
*lstm_cell_1_matmul_readvariableop_resource:	Θ?
,lstm_cell_1_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_1_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_1/BiasAdd/ReadVariableOp’!lstm_cell_1/MatMul/ReadVariableOp’#lstm_cell_1/MatMul_1/ReadVariableOp’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????ΰ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39376*
condR
while_cond_39375*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

Ό
-__inference_backward_lstm_layer_call_fn_43028
inputs_0
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity’StatefulPartitionedCallμ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_38801o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
Π"
Υ
while_body_38539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_2_38563_0:	Θ,
while_lstm_cell_2_38565_0:	2Θ(
while_lstm_cell_2_38567_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_2_38563:	Θ*
while_lstm_cell_2_38565:	2Θ&
while_lstm_cell_2_38567:	Θ’)while/lstm_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ͺ
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_38563_0while_lstm_cell_2_38565_0while_lstm_cell_2_38567_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_38525Ϋ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????2x

while/NoOpNoOp*^while/lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_2_38563while_lstm_cell_2_38563_0"4
while_lstm_cell_2_38565while_lstm_cell_2_38565_0"4
while_lstm_cell_2_38567while_lstm_cell_2_38567_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 

»
,__inference_forward_lstm_layer_call_fn_42401
inputs_0
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_forward_lstm_layer_call_and_return_conditional_losses_38256o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
η
τ
+__inference_lstm_cell_2_layer_call_fn_43762

inputs
states_0
states_1
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity

identity_1

identity_2’StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_38671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
G
ζ
forward_lstm_while_body_418026
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_35
1forward_lstm_while_forward_lstm_strided_slice_1_0q
mforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0R
?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘT
Aforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘO
@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
forward_lstm_while_identity!
forward_lstm_while_identity_1!
forward_lstm_while_identity_2!
forward_lstm_while_identity_3!
forward_lstm_while_identity_4!
forward_lstm_while_identity_53
/forward_lstm_while_forward_lstm_strided_slice_1o
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensorP
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	ΘR
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘM
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp
Dforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   η
6forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0forward_lstm_while_placeholderMforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0΅
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0ί
%forward_lstm/while/lstm_cell_1/MatMulMatMul=forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΉ
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ζ
'forward_lstm/while/lstm_cell_1/MatMul_1MatMul forward_lstm_while_placeholder_2>forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΒ
"forward_lstm/while/lstm_cell_1/addAddV2/forward_lstm/while/lstm_cell_1/MatMul:product:01forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ³
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Λ
&forward_lstm/while/lstm_cell_1/BiasAddBiasAdd&forward_lstm/while/lstm_cell_1/add:z:0=forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θp
.forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$forward_lstm/while/lstm_cell_1/splitSplit7forward_lstm/while/lstm_cell_1/split/split_dim:output:0/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
&forward_lstm/while/lstm_cell_1/SigmoidSigmoid-forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_1Sigmoid-forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2«
"forward_lstm/while/lstm_cell_1/mulMul,forward_lstm/while/lstm_cell_1/Sigmoid_1:y:0 forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
#forward_lstm/while/lstm_cell_1/ReluRelu-forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2Ό
$forward_lstm/while/lstm_cell_1/mul_1Mul*forward_lstm/while/lstm_cell_1/Sigmoid:y:01forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2±
$forward_lstm/while/lstm_cell_1/add_1AddV2&forward_lstm/while/lstm_cell_1/mul:z:0(forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_2Sigmoid-forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
%forward_lstm/while/lstm_cell_1/Relu_1Relu(forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2ΐ
$forward_lstm/while/lstm_cell_1/mul_2Mul,forward_lstm/while/lstm_cell_1/Sigmoid_2:y:03forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ψ
7forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem forward_lstm_while_placeholder_1forward_lstm_while_placeholder(forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?Z
forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/addAddV2forward_lstm_while_placeholder!forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: \
forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/add_1AddV22forward_lstm_while_forward_lstm_while_loop_counter#forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_lstm/while/IdentityIdentityforward_lstm/while/add_1:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_1Identity8forward_lstm_while_forward_lstm_while_maximum_iterations^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_2Identityforward_lstm/while/add:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: ΐ
forward_lstm/while/Identity_3IdentityGforward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?
forward_lstm/while/Identity_4Identity(forward_lstm/while/lstm_cell_1/mul_2:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/Identity_5Identity(forward_lstm/while/lstm_cell_1/add_1:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/NoOpNoOp6^forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5^forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp7^forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/forward_lstm_while_forward_lstm_strided_slice_11forward_lstm_while_forward_lstm_strided_slice_1_0"C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0"G
forward_lstm_while_identity_1&forward_lstm/while/Identity_1:output:0"G
forward_lstm_while_identity_2&forward_lstm/while/Identity_2:output:0"G
forward_lstm_while_identity_3&forward_lstm/while/Identity_3:output:0"G
forward_lstm_while_identity_4&forward_lstm/while/Identity_4:output:0"G
forward_lstm_while_identity_5&forward_lstm/while/Identity_5:output:0"
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0"
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"ά
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensormforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2n
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2l
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2p
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
γ7
Ζ
while_body_38875
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_1_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_1/BiasAdd/ReadVariableOp’'while/lstm_cell_1/MatMul/ReadVariableOp’)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????―
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ε	
σ
B__inference_dense_1_layer_call_and_return_conditional_losses_39852

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
ς΄

H__inference_bidirectional_layer_call_and_return_conditional_losses_40236

inputsJ
7forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	ΘL
9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘG
8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	ΘK
8backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	ΘM
:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘH
9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’backward_lstm/while’/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’forward_lstm/whileH
forward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:j
 forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_sliceStridedSliceforward_lstm/Shape:output:0)forward_lstm/strided_slice/stack:output:0+forward_lstm/strided_slice/stack_1:output:0+forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros/packedPack#forward_lstm/strided_slice:output:0$forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zerosFill"forward_lstm/zeros/packed:output:0!forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2_
forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros_1/packedPack#forward_lstm/strided_slice:output:0&forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zeros_1Fill$forward_lstm/zeros_1/packed:output:0#forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2p
forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_lstm/transpose	Transposeinputs$forward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????^
forward_lstm/Shape_1Shapeforward_lstm/transpose:y:0*
T0*
_output_shapes
:l
"forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_slice_1StridedSliceforward_lstm/Shape_1:output:0+forward_lstm/strided_slice_1/stack:output:0-forward_lstm/strided_slice_1/stack_1:output:0-forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????Ϋ
forward_lstm/TensorArrayV2TensorListReserve1forward_lstm/TensorArrayV2/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
Bforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
4forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm/transpose:y:0Kforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?l
"forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ͺ
forward_lstm/strided_slice_2StridedSliceforward_lstm/transpose:y:0+forward_lstm/strided_slice_2/stack:output:0-forward_lstm/strided_slice_2/stack_1:output:0-forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask§
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp7forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0»
forward_lstm/lstm_cell_1/MatMulMatMul%forward_lstm/strided_slice_2:output:06forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ«
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0΅
!forward_lstm/lstm_cell_1/MatMul_1MatMulforward_lstm/zeros:output:08forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ°
forward_lstm/lstm_cell_1/addAddV2)forward_lstm/lstm_cell_1/MatMul:product:0+forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ₯
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ή
 forward_lstm/lstm_cell_1/BiasAddBiasAdd forward_lstm/lstm_cell_1/add:z:07forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θj
(forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/lstm_cell_1/splitSplit1forward_lstm/lstm_cell_1/split/split_dim:output:0)forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
 forward_lstm/lstm_cell_1/SigmoidSigmoid'forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid'forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/mulMul&forward_lstm/lstm_cell_1/Sigmoid_1:y:0forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/ReluRelu'forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2ͺ
forward_lstm/lstm_cell_1/mul_1Mul$forward_lstm/lstm_cell_1/Sigmoid:y:0+forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/add_1AddV2 forward_lstm/lstm_cell_1/mul:z:0"forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid'forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2}
forward_lstm/lstm_cell_1/Relu_1Relu"forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2?
forward_lstm/lstm_cell_1/mul_2Mul&forward_lstm/lstm_cell_1/Sigmoid_2:y:0-forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2{
*forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ί
forward_lstm/TensorArrayV2_1TensorListReserve3forward_lstm/TensorArrayV2_1/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?S
forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
forward_lstm/whileWhile(forward_lstm/while/loop_counter:output:0.forward_lstm/while/maximum_iterations:output:0forward_lstm/time:output:0%forward_lstm/TensorArrayV2_1:handle:0forward_lstm/zeros:output:0forward_lstm/zeros_1:output:0%forward_lstm/strided_slice_1:output:0Dforward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07forward_lstm_lstm_cell_1_matmul_readvariableop_resource9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
forward_lstm_while_body_40009*)
cond!R
forward_lstm_while_cond_40008*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
=forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ι
/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm/while:output:3Fforward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0u
"forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Θ
forward_lstm/strided_slice_3StridedSlice8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+forward_lstm/strided_slice_3/stack:output:0-forward_lstm/strided_slice_3/stack_1:output:0-forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maskr
forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
forward_lstm/transpose_1	Transpose8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&forward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2h
forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    I
backward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:k
!backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_lstm/strided_sliceStridedSlicebackward_lstm/Shape:output:0*backward_lstm/strided_slice/stack:output:0,backward_lstm/strided_slice/stack_1:output:0,backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm/zeros/packedPack$backward_lstm/strided_slice:output:0%backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zerosFill#backward_lstm/zeros/packed:output:0"backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2`
backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2‘
backward_lstm/zeros_1/packedPack$backward_lstm/strided_slice:output:0'backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:`
backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zeros_1Fill%backward_lstm/zeros_1/packed:output:0$backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2q
backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_lstm/transpose	Transposeinputs%backward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????`
backward_lstm/Shape_1Shapebackward_lstm/transpose:y:0*
T0*
_output_shapes
:m
#backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:‘
backward_lstm/strided_slice_1StridedSlicebackward_lstm/Shape_1:output:0,backward_lstm/strided_slice_1/stack:output:0.backward_lstm/strided_slice_1/stack_1:output:0.backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ή
backward_lstm/TensorArrayV2TensorListReserve2backward_lstm/TensorArrayV2/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?f
backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
backward_lstm/ReverseV2	ReverseV2backward_lstm/transpose:y:0%backward_lstm/ReverseV2/axis:output:0*
T0*+
_output_shapes
:P?????????
Cbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
5backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor backward_lstm/ReverseV2:output:0Lbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?m
#backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:―
backward_lstm/strided_slice_2StridedSlicebackward_lstm/transpose:y:0,backward_lstm/strided_slice_2/stack:output:0.backward_lstm/strided_slice_2/stack_1:output:0.backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask©
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp8backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0Ύ
 backward_lstm/lstm_cell_2/MatMulMatMul&backward_lstm/strided_slice_2:output:07backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ­
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0Έ
"backward_lstm/lstm_cell_2/MatMul_1MatMulbackward_lstm/zeros:output:09backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ³
backward_lstm/lstm_cell_2/addAddV2*backward_lstm/lstm_cell_2/MatMul:product:0,backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ§
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ό
!backward_lstm/lstm_cell_2/BiasAddBiasAdd!backward_lstm/lstm_cell_2/add:z:08backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θk
)backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/lstm_cell_2/splitSplit2backward_lstm/lstm_cell_2/split/split_dim:output:0*backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
!backward_lstm/lstm_cell_2/SigmoidSigmoid(backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_1Sigmoid(backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/mulMul'backward_lstm/lstm_cell_2/Sigmoid_1:y:0backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/ReluRelu(backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2­
backward_lstm/lstm_cell_2/mul_1Mul%backward_lstm/lstm_cell_2/Sigmoid:y:0,backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2’
backward_lstm/lstm_cell_2/add_1AddV2!backward_lstm/lstm_cell_2/mul:z:0#backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_2Sigmoid(backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
 backward_lstm/lstm_cell_2/Relu_1Relu#backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2±
backward_lstm/lstm_cell_2/mul_2Mul'backward_lstm/lstm_cell_2/Sigmoid_2:y:0.backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2|
+backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   β
backward_lstm/TensorArrayV2_1TensorListReserve4backward_lstm/TensorArrayV2_1/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?T
backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Α
backward_lstm/whileWhile)backward_lstm/while/loop_counter:output:0/backward_lstm/while/maximum_iterations:output:0backward_lstm/time:output:0&backward_lstm/TensorArrayV2_1:handle:0backward_lstm/zeros:output:0backward_lstm/zeros_1:output:0&backward_lstm/strided_slice_1:output:0Ebackward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08backward_lstm_lstm_cell_2_matmul_readvariableop_resource:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
backward_lstm_while_body_40150**
cond"R 
backward_lstm_while_cond_40149*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
>backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   μ
0backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm/while:output:3Gbackward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0v
#backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ν
backward_lstm/strided_slice_3StridedSlice9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0,backward_lstm/strided_slice_3/stack:output:0.backward_lstm/strided_slice_3/stack_1:output:0.backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_masks
backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ΐ
backward_lstm/transpose_1	Transpose9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0'backward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2i
backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concatConcatV2%forward_lstm/strided_slice_3:output:0&backward_lstm/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????d 
NoOpNoOp1^backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0^backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2^backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp^backward_lstm/while0^forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/^forward_lstm/lstm_cell_1/MatMul/ReadVariableOp1^forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp^forward_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : : : 2d
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2b
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2f
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2*
backward_lstm/whilebackward_lstm/while2b
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2`
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2d
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2(
forward_lstm/whileforward_lstm/while:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
άX
Κ
,bidirectional_backward_lstm_while_body_40669T
Pbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_loop_counterZ
Vbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_maximum_iterations1
-bidirectional_backward_lstm_while_placeholder3
/bidirectional_backward_lstm_while_placeholder_13
/bidirectional_backward_lstm_while_placeholder_23
/bidirectional_backward_lstm_while_placeholder_3S
Obidirectional_backward_lstm_while_bidirectional_backward_lstm_strided_slice_1_0
bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0a
Nbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	Θc
Pbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2Θ^
Obidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ.
*bidirectional_backward_lstm_while_identity0
,bidirectional_backward_lstm_while_identity_10
,bidirectional_backward_lstm_while_identity_20
,bidirectional_backward_lstm_while_identity_30
,bidirectional_backward_lstm_while_identity_40
,bidirectional_backward_lstm_while_identity_5Q
Mbidirectional_backward_lstm_while_bidirectional_backward_lstm_strided_slice_1
bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_
Lbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	Θa
Nbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θ\
Mbidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’Dbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’Cbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’Ebidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp€
Sbidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ³
Ebidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0-bidirectional_backward_lstm_while_placeholder\bidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0Σ
Cbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpNbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0
4bidirectional/backward_lstm/while/lstm_cell_2/MatMulMatMulLbidirectional/backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Kbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΧ
Ebidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpPbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0σ
6bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1MatMul/bidirectional_backward_lstm_while_placeholder_2Mbidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θο
1bidirectional/backward_lstm/while/lstm_cell_2/addAddV2>bidirectional/backward_lstm/while/lstm_cell_2/MatMul:product:0@bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΡ
Dbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpObidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0ψ
5bidirectional/backward_lstm/while/lstm_cell_2/BiasAddBiasAdd5bidirectional/backward_lstm/while/lstm_cell_2/add:z:0Lbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
=bidirectional/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ΐ
3bidirectional/backward_lstm/while/lstm_cell_2/splitSplitFbidirectional/backward_lstm/while/lstm_cell_2/split/split_dim:output:0>bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split°
5bidirectional/backward_lstm/while/lstm_cell_2/SigmoidSigmoid<bidirectional/backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2²
7bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_1Sigmoid<bidirectional/backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2Ψ
1bidirectional/backward_lstm/while/lstm_cell_2/mulMul;bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0/bidirectional_backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2ͺ
2bidirectional/backward_lstm/while/lstm_cell_2/ReluRelu<bidirectional/backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2ι
3bidirectional/backward_lstm/while/lstm_cell_2/mul_1Mul9bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid:y:0@bidirectional/backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2ή
3bidirectional/backward_lstm/while/lstm_cell_2/add_1AddV25bidirectional/backward_lstm/while/lstm_cell_2/mul:z:07bidirectional/backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2²
7bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_2Sigmoid<bidirectional/backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2§
4bidirectional/backward_lstm/while/lstm_cell_2/Relu_1Relu7bidirectional/backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2ν
3bidirectional/backward_lstm/while/lstm_cell_2/mul_2Mul;bidirectional/backward_lstm/while/lstm_cell_2/Sigmoid_2:y:0Bbidirectional/backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2΄
Fbidirectional/backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/bidirectional_backward_lstm_while_placeholder_1-bidirectional_backward_lstm_while_placeholder7bidirectional/backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?i
'bidirectional/backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :°
%bidirectional/backward_lstm/while/addAddV2-bidirectional_backward_lstm_while_placeholder0bidirectional/backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: k
)bidirectional/backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Χ
'bidirectional/backward_lstm/while/add_1AddV2Pbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_loop_counter2bidirectional/backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: ­
*bidirectional/backward_lstm/while/IdentityIdentity+bidirectional/backward_lstm/while/add_1:z:0'^bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: Ϊ
,bidirectional/backward_lstm/while/Identity_1IdentityVbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_maximum_iterations'^bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: ­
,bidirectional/backward_lstm/while/Identity_2Identity)bidirectional/backward_lstm/while/add:z:0'^bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: ν
,bidirectional/backward_lstm/while/Identity_3IdentityVbidirectional/backward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^bidirectional/backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?Μ
,bidirectional/backward_lstm/while/Identity_4Identity7bidirectional/backward_lstm/while/lstm_cell_2/mul_2:z:0'^bidirectional/backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2Μ
,bidirectional/backward_lstm/while/Identity_5Identity7bidirectional/backward_lstm/while/lstm_cell_2/add_1:z:0'^bidirectional/backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2½
&bidirectional/backward_lstm/while/NoOpNoOpE^bidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpD^bidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpF^bidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 " 
Mbidirectional_backward_lstm_while_bidirectional_backward_lstm_strided_slice_1Obidirectional_backward_lstm_while_bidirectional_backward_lstm_strided_slice_1_0"a
*bidirectional_backward_lstm_while_identity3bidirectional/backward_lstm/while/Identity:output:0"e
,bidirectional_backward_lstm_while_identity_15bidirectional/backward_lstm/while/Identity_1:output:0"e
,bidirectional_backward_lstm_while_identity_25bidirectional/backward_lstm/while/Identity_2:output:0"e
,bidirectional_backward_lstm_while_identity_35bidirectional/backward_lstm/while/Identity_3:output:0"e
,bidirectional_backward_lstm_while_identity_45bidirectional/backward_lstm/while/Identity_4:output:0"e
,bidirectional_backward_lstm_while_identity_55bidirectional/backward_lstm/while/Identity_5:output:0" 
Mbidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceObidirectional_backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"’
Nbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resourcePbidirectional_backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
Lbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resourceNbidirectional_backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"
bidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensorbidirectional_backward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2
Dbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpDbidirectional/backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2
Cbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpCbidirectional/backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2
Ebidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpEbidirectional/backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 

Β
forward_lstm_while_cond_418016
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_38
4forward_lstm_while_less_forward_lstm_strided_slice_1M
Iforward_lstm_while_forward_lstm_while_cond_41801___redundant_placeholder0M
Iforward_lstm_while_forward_lstm_while_cond_41801___redundant_placeholder1M
Iforward_lstm_while_forward_lstm_while_cond_41801___redundant_placeholder2M
Iforward_lstm_while_forward_lstm_while_cond_41801___redundant_placeholder3
forward_lstm_while_identity

forward_lstm/while/LessLessforward_lstm_while_placeholder4forward_lstm_while_less_forward_lstm_strided_slice_1*
T0*
_output_shapes
: e
forward_lstm/while/IdentityIdentityforward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ϋ

F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_43794

inputs
states_0
states_11
matmul_readvariableop_resource:	Θ3
 matmul_1_readvariableop_resource:	2Θ.
biasadd_readvariableop_resource:	Θ
identity

identity_1

identity_2’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ά
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
Ϊ7
Ζ
while_body_42636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_1_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_1/BiasAdd/ReadVariableOp’'while/lstm_cell_1/MatMul/ReadVariableOp’)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ϊ7
Ζ
while_body_42493
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_1_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_1/BiasAdd/ReadVariableOp’'while/lstm_cell_1/MatMul/ReadVariableOp’)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
χ7

G__inference_forward_lstm_layer_call_and_return_conditional_losses_38447

inputs$
lstm_cell_1_38365:	Θ$
lstm_cell_1_38367:	2Θ 
lstm_cell_1_38369:	Θ
identity’#lstm_cell_1/StatefulPartitionedCall’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ΰ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskμ
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_38365lstm_cell_1_38367lstm_cell_1_38369*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_38319n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ―
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_38365lstm_cell_1_38367lstm_cell_1_38369*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38378*
condR
while_cond_38377*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2t
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
πη
§
E__inference_sequential_layer_call_and_return_conditional_losses_40769

inputsX
Ebidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	ΘZ
Gbidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘU
Fbidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	ΘY
Fbidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	Θ[
Hbidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘV
Gbidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	Θ6
$dense_matmul_readvariableop_resource:dP3
%dense_biasadd_readvariableop_resource:P8
&dense_1_matmul_readvariableop_resource:PP5
'dense_1_biasadd_readvariableop_resource:P
identity’>bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’=bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’?bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’!bidirectional/backward_lstm/while’=bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’<bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’>bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’ bidirectional/forward_lstm/while’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOpV
 bidirectional/forward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:x
.bidirectional/forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0bidirectional/forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0bidirectional/forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ψ
(bidirectional/forward_lstm/strided_sliceStridedSlice)bidirectional/forward_lstm/Shape:output:07bidirectional/forward_lstm/strided_slice/stack:output:09bidirectional/forward_lstm/strided_slice/stack_1:output:09bidirectional/forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)bidirectional/forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Δ
'bidirectional/forward_lstm/zeros/packedPack1bidirectional/forward_lstm/strided_slice:output:02bidirectional/forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&bidirectional/forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
 bidirectional/forward_lstm/zerosFill0bidirectional/forward_lstm/zeros/packed:output:0/bidirectional/forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2m
+bidirectional/forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Θ
)bidirectional/forward_lstm/zeros_1/packedPack1bidirectional/forward_lstm/strided_slice:output:04bidirectional/forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:m
(bidirectional/forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Γ
"bidirectional/forward_lstm/zeros_1Fill2bidirectional/forward_lstm/zeros_1/packed:output:01bidirectional/forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2~
)bidirectional/forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
$bidirectional/forward_lstm/transpose	Transposeinputs2bidirectional/forward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????z
"bidirectional/forward_lstm/Shape_1Shape(bidirectional/forward_lstm/transpose:y:0*
T0*
_output_shapes
:z
0bidirectional/forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2bidirectional/forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:β
*bidirectional/forward_lstm/strided_slice_1StridedSlice+bidirectional/forward_lstm/Shape_1:output:09bidirectional/forward_lstm/strided_slice_1/stack:output:0;bidirectional/forward_lstm/strided_slice_1/stack_1:output:0;bidirectional/forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6bidirectional/forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????
(bidirectional/forward_lstm/TensorArrayV2TensorListReserve?bidirectional/forward_lstm/TensorArrayV2/element_shape:output:03bidirectional/forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?‘
Pbidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ±
Bbidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(bidirectional/forward_lstm/transpose:y:0Ybidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?z
0bidirectional/forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2bidirectional/forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:π
*bidirectional/forward_lstm/strided_slice_2StridedSlice(bidirectional/forward_lstm/transpose:y:09bidirectional/forward_lstm/strided_slice_2/stack:output:0;bidirectional/forward_lstm/strided_slice_2/stack_1:output:0;bidirectional/forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskΓ
<bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpEbidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0ε
-bidirectional/forward_lstm/lstm_cell_1/MatMulMatMul3bidirectional/forward_lstm/strided_slice_2:output:0Dbidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΗ
>bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGbidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0ί
/bidirectional/forward_lstm/lstm_cell_1/MatMul_1MatMul)bidirectional/forward_lstm/zeros:output:0Fbidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΪ
*bidirectional/forward_lstm/lstm_cell_1/addAddV27bidirectional/forward_lstm/lstm_cell_1/MatMul:product:09bidirectional/forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΑ
=bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpFbidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0γ
.bidirectional/forward_lstm/lstm_cell_1/BiasAddBiasAdd.bidirectional/forward_lstm/lstm_cell_1/add:z:0Ebidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θx
6bidirectional/forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :«
,bidirectional/forward_lstm/lstm_cell_1/splitSplit?bidirectional/forward_lstm/lstm_cell_1/split/split_dim:output:07bidirectional/forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split’
.bidirectional/forward_lstm/lstm_cell_1/SigmoidSigmoid5bidirectional/forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2€
0bidirectional/forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid5bidirectional/forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2Ζ
*bidirectional/forward_lstm/lstm_cell_1/mulMul4bidirectional/forward_lstm/lstm_cell_1/Sigmoid_1:y:0+bidirectional/forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
+bidirectional/forward_lstm/lstm_cell_1/ReluRelu5bidirectional/forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2Τ
,bidirectional/forward_lstm/lstm_cell_1/mul_1Mul2bidirectional/forward_lstm/lstm_cell_1/Sigmoid:y:09bidirectional/forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2Ι
,bidirectional/forward_lstm/lstm_cell_1/add_1AddV2.bidirectional/forward_lstm/lstm_cell_1/mul:z:00bidirectional/forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2€
0bidirectional/forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid5bidirectional/forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
-bidirectional/forward_lstm/lstm_cell_1/Relu_1Relu0bidirectional/forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2Ψ
,bidirectional/forward_lstm/lstm_cell_1/mul_2Mul4bidirectional/forward_lstm/lstm_cell_1/Sigmoid_2:y:0;bidirectional/forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
8bidirectional/forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   
*bidirectional/forward_lstm/TensorArrayV2_1TensorListReserveAbidirectional/forward_lstm/TensorArrayV2_1/element_shape:output:03bidirectional/forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?a
bidirectional/forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3bidirectional/forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
-bidirectional/forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : χ
 bidirectional/forward_lstm/whileWhile6bidirectional/forward_lstm/while/loop_counter:output:0<bidirectional/forward_lstm/while/maximum_iterations:output:0(bidirectional/forward_lstm/time:output:03bidirectional/forward_lstm/TensorArrayV2_1:handle:0)bidirectional/forward_lstm/zeros:output:0+bidirectional/forward_lstm/zeros_1:output:03bidirectional/forward_lstm/strided_slice_1:output:0Rbidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ebidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resourceGbidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resourceFbidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *7
body/R-
+bidirectional_forward_lstm_while_body_40528*7
cond/R-
+bidirectional_forward_lstm_while_cond_40527*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
Kbidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   
=bidirectional/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStack)bidirectional/forward_lstm/while:output:3Tbidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0
0bidirectional/forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
2bidirectional/forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
*bidirectional/forward_lstm/strided_slice_3StridedSliceFbidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:09bidirectional/forward_lstm/strided_slice_3/stack:output:0;bidirectional/forward_lstm/strided_slice_3/stack_1:output:0;bidirectional/forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask
+bidirectional/forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          η
&bidirectional/forward_lstm/transpose_1	TransposeFbidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:04bidirectional/forward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2v
"bidirectional/forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    W
!bidirectional/backward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:y
/bidirectional/backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1bidirectional/backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1bidirectional/backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:έ
)bidirectional/backward_lstm/strided_sliceStridedSlice*bidirectional/backward_lstm/Shape:output:08bidirectional/backward_lstm/strided_slice/stack:output:0:bidirectional/backward_lstm/strided_slice/stack_1:output:0:bidirectional/backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*bidirectional/backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Η
(bidirectional/backward_lstm/zeros/packedPack2bidirectional/backward_lstm/strided_slice:output:03bidirectional/backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'bidirectional/backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ΐ
!bidirectional/backward_lstm/zerosFill1bidirectional/backward_lstm/zeros/packed:output:00bidirectional/backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2n
,bidirectional/backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Λ
*bidirectional/backward_lstm/zeros_1/packedPack2bidirectional/backward_lstm/strided_slice:output:05bidirectional/backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:n
)bidirectional/backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ζ
#bidirectional/backward_lstm/zeros_1Fill3bidirectional/backward_lstm/zeros_1/packed:output:02bidirectional/backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
*bidirectional/backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ₯
%bidirectional/backward_lstm/transpose	Transposeinputs3bidirectional/backward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????|
#bidirectional/backward_lstm/Shape_1Shape)bidirectional/backward_lstm/transpose:y:0*
T0*
_output_shapes
:{
1bidirectional/backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3bidirectional/backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3bidirectional/backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:η
+bidirectional/backward_lstm/strided_slice_1StridedSlice,bidirectional/backward_lstm/Shape_1:output:0:bidirectional/backward_lstm/strided_slice_1/stack:output:0<bidirectional/backward_lstm/strided_slice_1/stack_1:output:0<bidirectional/backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7bidirectional/backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????
)bidirectional/backward_lstm/TensorArrayV2TensorListReserve@bidirectional/backward_lstm/TensorArrayV2/element_shape:output:04bidirectional/backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?t
*bidirectional/backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Θ
%bidirectional/backward_lstm/ReverseV2	ReverseV2)bidirectional/backward_lstm/transpose:y:03bidirectional/backward_lstm/ReverseV2/axis:output:0*
T0*+
_output_shapes
:P?????????’
Qbidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   Ή
Cbidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor.bidirectional/backward_lstm/ReverseV2:output:0Zbidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?{
1bidirectional/backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3bidirectional/backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3bidirectional/backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:υ
+bidirectional/backward_lstm/strided_slice_2StridedSlice)bidirectional/backward_lstm/transpose:y:0:bidirectional/backward_lstm/strided_slice_2/stack:output:0<bidirectional/backward_lstm/strided_slice_2/stack_1:output:0<bidirectional/backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskΕ
=bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpFbidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0θ
.bidirectional/backward_lstm/lstm_cell_2/MatMulMatMul4bidirectional/backward_lstm/strided_slice_2:output:0Ebidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΙ
?bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpHbidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0β
0bidirectional/backward_lstm/lstm_cell_2/MatMul_1MatMul*bidirectional/backward_lstm/zeros:output:0Gbidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θέ
+bidirectional/backward_lstm/lstm_cell_2/addAddV28bidirectional/backward_lstm/lstm_cell_2/MatMul:product:0:bidirectional/backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΓ
>bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpGbidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0ζ
/bidirectional/backward_lstm/lstm_cell_2/BiasAddBiasAdd/bidirectional/backward_lstm/lstm_cell_2/add:z:0Fbidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
7bidirectional/backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-bidirectional/backward_lstm/lstm_cell_2/splitSplit@bidirectional/backward_lstm/lstm_cell_2/split/split_dim:output:08bidirectional/backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split€
/bidirectional/backward_lstm/lstm_cell_2/SigmoidSigmoid6bidirectional/backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2¦
1bidirectional/backward_lstm/lstm_cell_2/Sigmoid_1Sigmoid6bidirectional/backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2Ι
+bidirectional/backward_lstm/lstm_cell_2/mulMul5bidirectional/backward_lstm/lstm_cell_2/Sigmoid_1:y:0,bidirectional/backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
,bidirectional/backward_lstm/lstm_cell_2/ReluRelu6bidirectional/backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2Χ
-bidirectional/backward_lstm/lstm_cell_2/mul_1Mul3bidirectional/backward_lstm/lstm_cell_2/Sigmoid:y:0:bidirectional/backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2Μ
-bidirectional/backward_lstm/lstm_cell_2/add_1AddV2/bidirectional/backward_lstm/lstm_cell_2/mul:z:01bidirectional/backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2¦
1bidirectional/backward_lstm/lstm_cell_2/Sigmoid_2Sigmoid6bidirectional/backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
.bidirectional/backward_lstm/lstm_cell_2/Relu_1Relu1bidirectional/backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2Ϋ
-bidirectional/backward_lstm/lstm_cell_2/mul_2Mul5bidirectional/backward_lstm/lstm_cell_2/Sigmoid_2:y:0<bidirectional/backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
9bidirectional/backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   
+bidirectional/backward_lstm/TensorArrayV2_1TensorListReserveBbidirectional/backward_lstm/TensorArrayV2_1/element_shape:output:04bidirectional/backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?b
 bidirectional/backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4bidirectional/backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????p
.bidirectional/backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
!bidirectional/backward_lstm/whileWhile7bidirectional/backward_lstm/while/loop_counter:output:0=bidirectional/backward_lstm/while/maximum_iterations:output:0)bidirectional/backward_lstm/time:output:04bidirectional/backward_lstm/TensorArrayV2_1:handle:0*bidirectional/backward_lstm/zeros:output:0,bidirectional/backward_lstm/zeros_1:output:04bidirectional/backward_lstm/strided_slice_1:output:0Sbidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fbidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resourceHbidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resourceGbidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *8
body0R.
,bidirectional_backward_lstm_while_body_40669*8
cond0R.
,bidirectional_backward_lstm_while_cond_40668*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
Lbidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   
>bidirectional/backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStack*bidirectional/backward_lstm/while:output:3Ubidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0
1bidirectional/backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
3bidirectional/backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3bidirectional/backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
+bidirectional/backward_lstm/strided_slice_3StridedSliceGbidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0:bidirectional/backward_lstm/strided_slice_3/stack:output:0<bidirectional/backward_lstm/strided_slice_3/stack_1:output:0<bidirectional/backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask
,bidirectional/backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          κ
'bidirectional/backward_lstm/transpose_1	TransposeGbidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:05bidirectional/backward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2w
#bidirectional/backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    [
bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :κ
bidirectional/concatConcatV23bidirectional/forward_lstm/strided_slice_3:output:04bidirectional/backward_lstm/strided_slice_3:output:0"bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype0
dense/MatMulMatMulbidirectional/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pa
activation/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Pm
dropout/IdentityIdentityactivation/Relu:activations:0*
T0*'
_output_shapes
:?????????P
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pg
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????P
NoOpNoOp?^bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp>^bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp@^bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp"^bidirectional/backward_lstm/while>^bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp=^bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp?^bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp!^bidirectional/forward_lstm/while^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 2
>bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp>bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2~
=bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp=bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2
?bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp?bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2F
!bidirectional/backward_lstm/while!bidirectional/backward_lstm/while2~
=bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp=bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2|
<bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp<bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2
>bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp>bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2D
 bidirectional/forward_lstm/while bidirectional/forward_lstm/while2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	

-__inference_bidirectional_layer_call_fn_41137
inputs_0
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_39491o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
°
Ύ
while_cond_42778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_42778___redundant_placeholder03
/while_while_cond_42778___redundant_placeholder13
/while_while_cond_42778___redundant_placeholder23
/while_while_cond_42778___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ϊ7
Ζ
while_body_43111
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_2_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_2/BiasAdd/ReadVariableOp’'while/lstm_cell_2/MatMul/ReadVariableOp’)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 


a
B__inference_dropout_layer_call_and_return_conditional_losses_42371

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed2????[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????Po
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????Pi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????PY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
γ7
Ζ
while_body_42922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_1_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_1/BiasAdd/ReadVariableOp’'while/lstm_cell_1/MatMul/ReadVariableOp’)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????―
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ή
Φ
backward_lstm_while_cond_416568
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_3:
6backward_lstm_while_less_backward_lstm_strided_slice_1O
Kbackward_lstm_while_backward_lstm_while_cond_41656___redundant_placeholder0O
Kbackward_lstm_while_backward_lstm_while_cond_41656___redundant_placeholder1O
Kbackward_lstm_while_backward_lstm_while_cond_41656___redundant_placeholder2O
Kbackward_lstm_while_backward_lstm_while_cond_41656___redundant_placeholder3 
backward_lstm_while_identity

backward_lstm/while/LessLessbackward_lstm_while_placeholder6backward_lstm_while_less_backward_lstm_strided_slice_1*
T0*
_output_shapes
: g
backward_lstm/while/IdentityIdentitybackward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
¦
Ϊ
+bidirectional_forward_lstm_while_cond_40527R
Nbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_loop_counterX
Tbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_maximum_iterations0
,bidirectional_forward_lstm_while_placeholder2
.bidirectional_forward_lstm_while_placeholder_12
.bidirectional_forward_lstm_while_placeholder_22
.bidirectional_forward_lstm_while_placeholder_3T
Pbidirectional_forward_lstm_while_less_bidirectional_forward_lstm_strided_slice_1i
ebidirectional_forward_lstm_while_bidirectional_forward_lstm_while_cond_40527___redundant_placeholder0i
ebidirectional_forward_lstm_while_bidirectional_forward_lstm_while_cond_40527___redundant_placeholder1i
ebidirectional_forward_lstm_while_bidirectional_forward_lstm_while_cond_40527___redundant_placeholder2i
ebidirectional_forward_lstm_while_bidirectional_forward_lstm_while_cond_40527___redundant_placeholder3-
)bidirectional_forward_lstm_while_identity
Ξ
%bidirectional/forward_lstm/while/LessLess,bidirectional_forward_lstm_while_placeholderPbidirectional_forward_lstm_while_less_bidirectional_forward_lstm_strided_slice_1*
T0*
_output_shapes
: 
)bidirectional/forward_lstm/while/IdentityIdentity)bidirectional/forward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "_
)bidirectional_forward_lstm_while_identity2bidirectional/forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ί

%__inference_dense_layer_call_fn_42324

inputs
unknown:dP
	unknown_0:P
identity’StatefulPartitionedCallΥ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_39822o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Γ	
ρ
@__inference_dense_layer_call_and_return_conditional_losses_39822

inputs0
matmul_readvariableop_resource:dP-
biasadd_readvariableop_resource:P
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ι
a
E__inference_activation_layer_call_and_return_conditional_losses_39833

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????PZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
Π"
Υ
while_body_38732
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_2_38756_0:	Θ,
while_lstm_cell_2_38758_0:	2Θ(
while_lstm_cell_2_38760_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_2_38756:	Θ*
while_lstm_cell_2_38758:	2Θ&
while_lstm_cell_2_38760:	Θ’)while/lstm_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ͺ
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_38756_0while_lstm_cell_2_38758_0while_lstm_cell_2_38760_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_38671Ϋ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????2x

while/NoOpNoOp*^while/lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_2_38756while_lstm_cell_2_38756_0"4
while_lstm_cell_2_38758while_lstm_cell_2_38758_0"4
while_lstm_cell_2_38760while_lstm_cell_2_38760_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
L

H__inference_backward_lstm_layer_call_and_return_conditional_losses_43195
inputs_0=
*lstm_cell_2_matmul_readvariableop_resource:	Θ?
,lstm_cell_2_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_2/BiasAdd/ReadVariableOp’!lstm_cell_2/MatMul/ReadVariableOp’#lstm_cell_2/MatMul_1/ReadVariableOp’while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ε
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_43111*
condR
while_cond_43110*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
°
Ύ
while_cond_39026
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_39026___redundant_placeholder03
/while_while_cond_39026___redundant_placeholder13
/while_while_cond_39026___redundant_placeholder23
/while_while_cond_39026___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
η
τ
+__inference_lstm_cell_1_layer_call_fn_43647

inputs
states_0
states_1
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity

identity_1

identity_2’StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_38173o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
Ά

H__inference_bidirectional_layer_call_and_return_conditional_losses_41457
inputs_0J
7forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	ΘL
9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘG
8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	ΘK
8backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	ΘM
:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘH
9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’backward_lstm/while’/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’forward_lstm/whileJ
forward_lstm/ShapeShapeinputs_0*
T0*
_output_shapes
:j
 forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_sliceStridedSliceforward_lstm/Shape:output:0)forward_lstm/strided_slice/stack:output:0+forward_lstm/strided_slice/stack_1:output:0+forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros/packedPack#forward_lstm/strided_slice:output:0$forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zerosFill"forward_lstm/zeros/packed:output:0!forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2_
forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros_1/packedPack#forward_lstm/strided_slice:output:0&forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zeros_1Fill$forward_lstm/zeros_1/packed:output:0#forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2p
forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_lstm/transpose	Transposeinputs_0$forward_lstm/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????^
forward_lstm/Shape_1Shapeforward_lstm/transpose:y:0*
T0*
_output_shapes
:l
"forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_slice_1StridedSliceforward_lstm/Shape_1:output:0+forward_lstm/strided_slice_1/stack:output:0-forward_lstm/strided_slice_1/stack_1:output:0-forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????Ϋ
forward_lstm/TensorArrayV2TensorListReserve1forward_lstm/TensorArrayV2/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
Bforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????
4forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm/transpose:y:0Kforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?l
"forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
forward_lstm/strided_slice_2StridedSliceforward_lstm/transpose:y:0+forward_lstm/strided_slice_2/stack:output:0-forward_lstm/strided_slice_2/stack_1:output:0-forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask§
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp7forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0»
forward_lstm/lstm_cell_1/MatMulMatMul%forward_lstm/strided_slice_2:output:06forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ«
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0΅
!forward_lstm/lstm_cell_1/MatMul_1MatMulforward_lstm/zeros:output:08forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ°
forward_lstm/lstm_cell_1/addAddV2)forward_lstm/lstm_cell_1/MatMul:product:0+forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ₯
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ή
 forward_lstm/lstm_cell_1/BiasAddBiasAdd forward_lstm/lstm_cell_1/add:z:07forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θj
(forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/lstm_cell_1/splitSplit1forward_lstm/lstm_cell_1/split/split_dim:output:0)forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
 forward_lstm/lstm_cell_1/SigmoidSigmoid'forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid'forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/mulMul&forward_lstm/lstm_cell_1/Sigmoid_1:y:0forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/ReluRelu'forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2ͺ
forward_lstm/lstm_cell_1/mul_1Mul$forward_lstm/lstm_cell_1/Sigmoid:y:0+forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/add_1AddV2 forward_lstm/lstm_cell_1/mul:z:0"forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid'forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2}
forward_lstm/lstm_cell_1/Relu_1Relu"forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2?
forward_lstm/lstm_cell_1/mul_2Mul&forward_lstm/lstm_cell_1/Sigmoid_2:y:0-forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2{
*forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ί
forward_lstm/TensorArrayV2_1TensorListReserve3forward_lstm/TensorArrayV2_1/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?S
forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
forward_lstm/whileWhile(forward_lstm/while/loop_counter:output:0.forward_lstm/while/maximum_iterations:output:0forward_lstm/time:output:0%forward_lstm/TensorArrayV2_1:handle:0forward_lstm/zeros:output:0forward_lstm/zeros_1:output:0%forward_lstm/strided_slice_1:output:0Dforward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07forward_lstm_lstm_cell_1_matmul_readvariableop_resource9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
forward_lstm_while_body_41230*)
cond!R
forward_lstm_while_cond_41229*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
=forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ς
/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm/while:output:3Fforward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0u
"forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Θ
forward_lstm/strided_slice_3StridedSlice8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+forward_lstm/strided_slice_3/stack:output:0-forward_lstm/strided_slice_3/stack_1:output:0-forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maskr
forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ζ
forward_lstm/transpose_1	Transpose8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&forward_lstm/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2h
forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    K
backward_lstm/ShapeShapeinputs_0*
T0*
_output_shapes
:k
!backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_lstm/strided_sliceStridedSlicebackward_lstm/Shape:output:0*backward_lstm/strided_slice/stack:output:0,backward_lstm/strided_slice/stack_1:output:0,backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm/zeros/packedPack$backward_lstm/strided_slice:output:0%backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zerosFill#backward_lstm/zeros/packed:output:0"backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2`
backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2‘
backward_lstm/zeros_1/packedPack$backward_lstm/strided_slice:output:0'backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:`
backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zeros_1Fill%backward_lstm/zeros_1/packed:output:0$backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2q
backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_lstm/transpose	Transposeinputs_0%backward_lstm/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
backward_lstm/Shape_1Shapebackward_lstm/transpose:y:0*
T0*
_output_shapes
:m
#backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:‘
backward_lstm/strided_slice_1StridedSlicebackward_lstm/Shape_1:output:0,backward_lstm/strided_slice_1/stack:output:0.backward_lstm/strided_slice_1/stack_1:output:0.backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ή
backward_lstm/TensorArrayV2TensorListReserve2backward_lstm/TensorArrayV2/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?f
backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: °
backward_lstm/ReverseV2	ReverseV2backward_lstm/transpose:y:0%backward_lstm/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????
Cbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????
5backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor backward_lstm/ReverseV2:output:0Lbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?m
#backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Έ
backward_lstm/strided_slice_2StridedSlicebackward_lstm/transpose:y:0,backward_lstm/strided_slice_2/stack:output:0.backward_lstm/strided_slice_2/stack_1:output:0.backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask©
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp8backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0Ύ
 backward_lstm/lstm_cell_2/MatMulMatMul&backward_lstm/strided_slice_2:output:07backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ­
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0Έ
"backward_lstm/lstm_cell_2/MatMul_1MatMulbackward_lstm/zeros:output:09backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ³
backward_lstm/lstm_cell_2/addAddV2*backward_lstm/lstm_cell_2/MatMul:product:0,backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ§
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ό
!backward_lstm/lstm_cell_2/BiasAddBiasAdd!backward_lstm/lstm_cell_2/add:z:08backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θk
)backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/lstm_cell_2/splitSplit2backward_lstm/lstm_cell_2/split/split_dim:output:0*backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
!backward_lstm/lstm_cell_2/SigmoidSigmoid(backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_1Sigmoid(backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/mulMul'backward_lstm/lstm_cell_2/Sigmoid_1:y:0backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/ReluRelu(backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2­
backward_lstm/lstm_cell_2/mul_1Mul%backward_lstm/lstm_cell_2/Sigmoid:y:0,backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2’
backward_lstm/lstm_cell_2/add_1AddV2!backward_lstm/lstm_cell_2/mul:z:0#backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_2Sigmoid(backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
 backward_lstm/lstm_cell_2/Relu_1Relu#backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2±
backward_lstm/lstm_cell_2/mul_2Mul'backward_lstm/lstm_cell_2/Sigmoid_2:y:0.backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2|
+backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   β
backward_lstm/TensorArrayV2_1TensorListReserve4backward_lstm/TensorArrayV2_1/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?T
backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Α
backward_lstm/whileWhile)backward_lstm/while/loop_counter:output:0/backward_lstm/while/maximum_iterations:output:0backward_lstm/time:output:0&backward_lstm/TensorArrayV2_1:handle:0backward_lstm/zeros:output:0backward_lstm/zeros_1:output:0&backward_lstm/strided_slice_1:output:0Ebackward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08backward_lstm_lstm_cell_2_matmul_readvariableop_resource:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
backward_lstm_while_body_41371**
cond"R 
backward_lstm_while_cond_41370*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
>backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   υ
0backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm/while:output:3Gbackward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0v
#backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ν
backward_lstm/strided_slice_3StridedSlice9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0,backward_lstm/strided_slice_3/stack:output:0.backward_lstm/strided_slice_3/stack_1:output:0.backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_masks
backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ι
backward_lstm/transpose_1	Transpose9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0'backward_lstm/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2i
backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concatConcatV2%forward_lstm/strided_slice_3:output:0&backward_lstm/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????d 
NoOpNoOp1^backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0^backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2^backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp^backward_lstm/while0^forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/^forward_lstm/lstm_cell_1/MatMul/ReadVariableOp1^forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp^forward_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2d
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2b
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2f
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2*
backward_lstm/whilebackward_lstm/while2b
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2`
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2d
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2(
forward_lstm/whileforward_lstm/while:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
€J

G__inference_forward_lstm_layer_call_and_return_conditional_losses_42720
inputs_0=
*lstm_cell_1_matmul_readvariableop_resource:	Θ?
,lstm_cell_1_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_1_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_1/BiasAdd/ReadVariableOp’!lstm_cell_1/MatMul/ReadVariableOp’#lstm_cell_1/MatMul_1/ReadVariableOp’while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ΰ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42636*
condR
while_cond_42635*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
η
τ
+__inference_lstm_cell_1_layer_call_fn_43664

inputs
states_0
states_1
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity

identity_1

identity_2’StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_38319o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
¦

ω
*__inference_sequential_layer_call_fn_40469

inputs
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
	unknown_5:dP
	unknown_6:P
	unknown_7:PP
	unknown_8:P
identity’StatefulPartitionedCallΒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_40307o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
L

H__inference_backward_lstm_layer_call_and_return_conditional_losses_43340
inputs_0=
*lstm_cell_2_matmul_readvariableop_resource:	Θ?
,lstm_cell_2_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_2/BiasAdd/ReadVariableOp’!lstm_cell_2/MatMul/ReadVariableOp’#lstm_cell_2/MatMul_1/ReadVariableOp’while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ε
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_43256*
condR
while_cond_43255*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
©L

H__inference_backward_lstm_layer_call_and_return_conditional_losses_43485

inputs=
*lstm_cell_2_matmul_readvariableop_resource:	Θ?
,lstm_cell_2_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_2/BiasAdd/ReadVariableOp’!lstm_cell_2/MatMul/ReadVariableOp’#lstm_cell_2/MatMul_1/ReadVariableOp’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????ε
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_43401*
condR
while_cond_43400*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ϋ

F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_43696

inputs
states_0
states_11
matmul_readvariableop_resource:	Θ3
 matmul_1_readvariableop_resource:	2Θ.
biasadd_readvariableop_resource:	Θ
identity

identity_1

identity_2’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ά
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
χ7

G__inference_forward_lstm_layer_call_and_return_conditional_losses_38256

inputs$
lstm_cell_1_38174:	Θ$
lstm_cell_1_38176:	2Θ 
lstm_cell_1_38178:	Θ
identity’#lstm_cell_1/StatefulPartitionedCall’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ΰ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskμ
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_38174lstm_cell_1_38176lstm_cell_1_38178*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_38173n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ―
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_38174lstm_cell_1_38176lstm_cell_1_38178*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38187*
condR
while_cond_38186*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2t
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
β

 __inference__wrapped_model_38106
bidirectional_inputc
Psequential_bidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	Θe
Rsequential_bidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ`
Qsequential_bidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	Θd
Qsequential_bidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	Θf
Ssequential_bidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2Θa
Rsequential_bidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	ΘA
/sequential_dense_matmul_readvariableop_resource:dP>
0sequential_dense_biasadd_readvariableop_resource:PC
1sequential_dense_1_matmul_readvariableop_resource:PP@
2sequential_dense_1_biasadd_readvariableop_resource:P
identity’Isequential/bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’Hsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’Jsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’,sequential/bidirectional/backward_lstm/while’Hsequential/bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’Gsequential/bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’Isequential/bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’+sequential/bidirectional/forward_lstm/while’'sequential/dense/BiasAdd/ReadVariableOp’&sequential/dense/MatMul/ReadVariableOp’)sequential/dense_1/BiasAdd/ReadVariableOp’(sequential/dense_1/MatMul/ReadVariableOpn
+sequential/bidirectional/forward_lstm/ShapeShapebidirectional_input*
T0*
_output_shapes
:
9sequential/bidirectional/forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;sequential/bidirectional/forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;sequential/bidirectional/forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3sequential/bidirectional/forward_lstm/strided_sliceStridedSlice4sequential/bidirectional/forward_lstm/Shape:output:0Bsequential/bidirectional/forward_lstm/strided_slice/stack:output:0Dsequential/bidirectional/forward_lstm/strided_slice/stack_1:output:0Dsequential/bidirectional/forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4sequential/bidirectional/forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2ε
2sequential/bidirectional/forward_lstm/zeros/packedPack<sequential/bidirectional/forward_lstm/strided_slice:output:0=sequential/bidirectional/forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1sequential/bidirectional/forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ή
+sequential/bidirectional/forward_lstm/zerosFill;sequential/bidirectional/forward_lstm/zeros/packed:output:0:sequential/bidirectional/forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2x
6sequential/bidirectional/forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2ι
4sequential/bidirectional/forward_lstm/zeros_1/packedPack<sequential/bidirectional/forward_lstm/strided_slice:output:0?sequential/bidirectional/forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:x
3sequential/bidirectional/forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    δ
-sequential/bidirectional/forward_lstm/zeros_1Fill=sequential/bidirectional/forward_lstm/zeros_1/packed:output:0<sequential/bidirectional/forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
4sequential/bidirectional/forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ζ
/sequential/bidirectional/forward_lstm/transpose	Transposebidirectional_input=sequential/bidirectional/forward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????
-sequential/bidirectional/forward_lstm/Shape_1Shape3sequential/bidirectional/forward_lstm/transpose:y:0*
T0*
_output_shapes
:
;sequential/bidirectional/forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential/bidirectional/forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential/bidirectional/forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/bidirectional/forward_lstm/strided_slice_1StridedSlice6sequential/bidirectional/forward_lstm/Shape_1:output:0Dsequential/bidirectional/forward_lstm/strided_slice_1/stack:output:0Fsequential/bidirectional/forward_lstm/strided_slice_1/stack_1:output:0Fsequential/bidirectional/forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Asequential/bidirectional/forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????¦
3sequential/bidirectional/forward_lstm/TensorArrayV2TensorListReserveJsequential/bidirectional/forward_lstm/TensorArrayV2/element_shape:output:0>sequential/bidirectional/forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?¬
[sequential/bidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Msequential/bidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor3sequential/bidirectional/forward_lstm/transpose:y:0dsequential/bidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
;sequential/bidirectional/forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential/bidirectional/forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential/bidirectional/forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:§
5sequential/bidirectional/forward_lstm/strided_slice_2StridedSlice3sequential/bidirectional/forward_lstm/transpose:y:0Dsequential/bidirectional/forward_lstm/strided_slice_2/stack:output:0Fsequential/bidirectional/forward_lstm/strided_slice_2/stack_1:output:0Fsequential/bidirectional/forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskΩ
Gsequential/bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpPsequential_bidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
8sequential/bidirectional/forward_lstm/lstm_cell_1/MatMulMatMul>sequential/bidirectional/forward_lstm/strided_slice_2:output:0Osequential/bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θέ
Isequential/bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpRsequential_bidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
:sequential/bidirectional/forward_lstm/lstm_cell_1/MatMul_1MatMul4sequential/bidirectional/forward_lstm/zeros:output:0Qsequential/bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θϋ
5sequential/bidirectional/forward_lstm/lstm_cell_1/addAddV2Bsequential/bidirectional/forward_lstm/lstm_cell_1/MatMul:product:0Dsequential/bidirectional/forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΧ
Hsequential/bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpQsequential_bidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
9sequential/bidirectional/forward_lstm/lstm_cell_1/BiasAddBiasAdd9sequential/bidirectional/forward_lstm/lstm_cell_1/add:z:0Psequential/bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
Asequential/bidirectional/forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Μ
7sequential/bidirectional/forward_lstm/lstm_cell_1/splitSplitJsequential/bidirectional/forward_lstm/lstm_cell_1/split/split_dim:output:0Bsequential/bidirectional/forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitΈ
9sequential/bidirectional/forward_lstm/lstm_cell_1/SigmoidSigmoid@sequential/bidirectional/forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2Ί
;sequential/bidirectional/forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid@sequential/bidirectional/forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2η
5sequential/bidirectional/forward_lstm/lstm_cell_1/mulMul?sequential/bidirectional/forward_lstm/lstm_cell_1/Sigmoid_1:y:06sequential/bidirectional/forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2²
6sequential/bidirectional/forward_lstm/lstm_cell_1/ReluRelu@sequential/bidirectional/forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2υ
7sequential/bidirectional/forward_lstm/lstm_cell_1/mul_1Mul=sequential/bidirectional/forward_lstm/lstm_cell_1/Sigmoid:y:0Dsequential/bidirectional/forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2κ
7sequential/bidirectional/forward_lstm/lstm_cell_1/add_1AddV29sequential/bidirectional/forward_lstm/lstm_cell_1/mul:z:0;sequential/bidirectional/forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2Ί
;sequential/bidirectional/forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid@sequential/bidirectional/forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2―
8sequential/bidirectional/forward_lstm/lstm_cell_1/Relu_1Relu;sequential/bidirectional/forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2ω
7sequential/bidirectional/forward_lstm/lstm_cell_1/mul_2Mul?sequential/bidirectional/forward_lstm/lstm_cell_1/Sigmoid_2:y:0Fsequential/bidirectional/forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
Csequential/bidirectional/forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ͺ
5sequential/bidirectional/forward_lstm/TensorArrayV2_1TensorListReserveLsequential/bidirectional/forward_lstm/TensorArrayV2_1/element_shape:output:0>sequential/bidirectional/forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?l
*sequential/bidirectional/forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 
>sequential/bidirectional/forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????z
8sequential/bidirectional/forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 

+sequential/bidirectional/forward_lstm/whileWhileAsequential/bidirectional/forward_lstm/while/loop_counter:output:0Gsequential/bidirectional/forward_lstm/while/maximum_iterations:output:03sequential/bidirectional/forward_lstm/time:output:0>sequential/bidirectional/forward_lstm/TensorArrayV2_1:handle:04sequential/bidirectional/forward_lstm/zeros:output:06sequential/bidirectional/forward_lstm/zeros_1:output:0>sequential/bidirectional/forward_lstm/strided_slice_1:output:0]sequential/bidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Psequential_bidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resourceRsequential_bidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resourceQsequential_bidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *B
body:R8
6sequential_bidirectional_forward_lstm_while_body_37865*B
cond:R8
6sequential_bidirectional_forward_lstm_while_cond_37864*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations §
Vsequential/bidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ΄
Hsequential/bidirectional/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStack4sequential/bidirectional/forward_lstm/while:output:3_sequential/bidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0
;sequential/bidirectional/forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????
=sequential/bidirectional/forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=sequential/bidirectional/forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ε
5sequential/bidirectional/forward_lstm/strided_slice_3StridedSliceQsequential/bidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0Dsequential/bidirectional/forward_lstm/strided_slice_3/stack:output:0Fsequential/bidirectional/forward_lstm/strided_slice_3/stack_1:output:0Fsequential/bidirectional/forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask
6sequential/bidirectional/forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
1sequential/bidirectional/forward_lstm/transpose_1	TransposeQsequential/bidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0?sequential/bidirectional/forward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2
-sequential/bidirectional/forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    o
,sequential/bidirectional/backward_lstm/ShapeShapebidirectional_input*
T0*
_output_shapes
:
:sequential/bidirectional/backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<sequential/bidirectional/backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<sequential/bidirectional/backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4sequential/bidirectional/backward_lstm/strided_sliceStridedSlice5sequential/bidirectional/backward_lstm/Shape:output:0Csequential/bidirectional/backward_lstm/strided_slice/stack:output:0Esequential/bidirectional/backward_lstm/strided_slice/stack_1:output:0Esequential/bidirectional/backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5sequential/bidirectional/backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2θ
3sequential/bidirectional/backward_lstm/zeros/packedPack=sequential/bidirectional/backward_lstm/strided_slice:output:0>sequential/bidirectional/backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:w
2sequential/bidirectional/backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    α
,sequential/bidirectional/backward_lstm/zerosFill<sequential/bidirectional/backward_lstm/zeros/packed:output:0;sequential/bidirectional/backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2y
7sequential/bidirectional/backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2μ
5sequential/bidirectional/backward_lstm/zeros_1/packedPack=sequential/bidirectional/backward_lstm/strided_slice:output:0@sequential/bidirectional/backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:y
4sequential/bidirectional/backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    η
.sequential/bidirectional/backward_lstm/zeros_1Fill>sequential/bidirectional/backward_lstm/zeros_1/packed:output:0=sequential/bidirectional/backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
5sequential/bidirectional/backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Θ
0sequential/bidirectional/backward_lstm/transpose	Transposebidirectional_input>sequential/bidirectional/backward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????
.sequential/bidirectional/backward_lstm/Shape_1Shape4sequential/bidirectional/backward_lstm/transpose:y:0*
T0*
_output_shapes
:
<sequential/bidirectional/backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>sequential/bidirectional/backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>sequential/bidirectional/backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6sequential/bidirectional/backward_lstm/strided_slice_1StridedSlice7sequential/bidirectional/backward_lstm/Shape_1:output:0Esequential/bidirectional/backward_lstm/strided_slice_1/stack:output:0Gsequential/bidirectional/backward_lstm/strided_slice_1/stack_1:output:0Gsequential/bidirectional/backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Bsequential/bidirectional/backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????©
4sequential/bidirectional/backward_lstm/TensorArrayV2TensorListReserveKsequential/bidirectional/backward_lstm/TensorArrayV2/element_shape:output:0?sequential/bidirectional/backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5sequential/bidirectional/backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ι
0sequential/bidirectional/backward_lstm/ReverseV2	ReverseV24sequential/bidirectional/backward_lstm/transpose:y:0>sequential/bidirectional/backward_lstm/ReverseV2/axis:output:0*
T0*+
_output_shapes
:P?????????­
\sequential/bidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   Ϊ
Nsequential/bidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor9sequential/bidirectional/backward_lstm/ReverseV2:output:0esequential/bidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
<sequential/bidirectional/backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>sequential/bidirectional/backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>sequential/bidirectional/backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
6sequential/bidirectional/backward_lstm/strided_slice_2StridedSlice4sequential/bidirectional/backward_lstm/transpose:y:0Esequential/bidirectional/backward_lstm/strided_slice_2/stack:output:0Gsequential/bidirectional/backward_lstm/strided_slice_2/stack_1:output:0Gsequential/bidirectional/backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskΫ
Hsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpQsequential_bidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
9sequential/bidirectional/backward_lstm/lstm_cell_2/MatMulMatMul?sequential/bidirectional/backward_lstm/strided_slice_2:output:0Psequential/bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θί
Jsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpSsequential_bidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
;sequential/bidirectional/backward_lstm/lstm_cell_2/MatMul_1MatMul5sequential/bidirectional/backward_lstm/zeros:output:0Rsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θώ
6sequential/bidirectional/backward_lstm/lstm_cell_2/addAddV2Csequential/bidirectional/backward_lstm/lstm_cell_2/MatMul:product:0Esequential/bidirectional/backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΩ
Isequential/bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpRsequential_bidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
:sequential/bidirectional/backward_lstm/lstm_cell_2/BiasAddBiasAdd:sequential/bidirectional/backward_lstm/lstm_cell_2/add:z:0Qsequential/bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
Bsequential/bidirectional/backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ο
8sequential/bidirectional/backward_lstm/lstm_cell_2/splitSplitKsequential/bidirectional/backward_lstm/lstm_cell_2/split/split_dim:output:0Csequential/bidirectional/backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitΊ
:sequential/bidirectional/backward_lstm/lstm_cell_2/SigmoidSigmoidAsequential/bidirectional/backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2Ό
<sequential/bidirectional/backward_lstm/lstm_cell_2/Sigmoid_1SigmoidAsequential/bidirectional/backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2κ
6sequential/bidirectional/backward_lstm/lstm_cell_2/mulMul@sequential/bidirectional/backward_lstm/lstm_cell_2/Sigmoid_1:y:07sequential/bidirectional/backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2΄
7sequential/bidirectional/backward_lstm/lstm_cell_2/ReluReluAsequential/bidirectional/backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2ψ
8sequential/bidirectional/backward_lstm/lstm_cell_2/mul_1Mul>sequential/bidirectional/backward_lstm/lstm_cell_2/Sigmoid:y:0Esequential/bidirectional/backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2ν
8sequential/bidirectional/backward_lstm/lstm_cell_2/add_1AddV2:sequential/bidirectional/backward_lstm/lstm_cell_2/mul:z:0<sequential/bidirectional/backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2Ό
<sequential/bidirectional/backward_lstm/lstm_cell_2/Sigmoid_2SigmoidAsequential/bidirectional/backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2±
9sequential/bidirectional/backward_lstm/lstm_cell_2/Relu_1Relu<sequential/bidirectional/backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2ό
8sequential/bidirectional/backward_lstm/lstm_cell_2/mul_2Mul@sequential/bidirectional/backward_lstm/lstm_cell_2/Sigmoid_2:y:0Gsequential/bidirectional/backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
Dsequential/bidirectional/backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ­
6sequential/bidirectional/backward_lstm/TensorArrayV2_1TensorListReserveMsequential/bidirectional/backward_lstm/TensorArrayV2_1/element_shape:output:0?sequential/bidirectional/backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?m
+sequential/bidirectional/backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 
?sequential/bidirectional/backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????{
9sequential/bidirectional/backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 

,sequential/bidirectional/backward_lstm/whileWhileBsequential/bidirectional/backward_lstm/while/loop_counter:output:0Hsequential/bidirectional/backward_lstm/while/maximum_iterations:output:04sequential/bidirectional/backward_lstm/time:output:0?sequential/bidirectional/backward_lstm/TensorArrayV2_1:handle:05sequential/bidirectional/backward_lstm/zeros:output:07sequential/bidirectional/backward_lstm/zeros_1:output:0?sequential/bidirectional/backward_lstm/strided_slice_1:output:0^sequential/bidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Qsequential_bidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resourceSsequential_bidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resourceRsequential_bidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *C
body;R9
7sequential_bidirectional_backward_lstm_while_body_38006*C
cond;R9
7sequential_bidirectional_backward_lstm_while_cond_38005*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations ¨
Wsequential/bidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ·
Isequential/bidirectional/backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStack5sequential/bidirectional/backward_lstm/while:output:3`sequential/bidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0
<sequential/bidirectional/backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????
>sequential/bidirectional/backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>sequential/bidirectional/backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Κ
6sequential/bidirectional/backward_lstm/strided_slice_3StridedSliceRsequential/bidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0Esequential/bidirectional/backward_lstm/strided_slice_3/stack:output:0Gsequential/bidirectional/backward_lstm/strided_slice_3/stack_1:output:0Gsequential/bidirectional/backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask
7sequential/bidirectional/backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
2sequential/bidirectional/backward_lstm/transpose_1	TransposeRsequential/bidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0@sequential/bidirectional/backward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2
.sequential/bidirectional/backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    f
$sequential/bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
sequential/bidirectional/concatConcatV2>sequential/bidirectional/forward_lstm/strided_slice_3:output:0?sequential/bidirectional/backward_lstm/strided_slice_3:output:0-sequential/bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype0­
sequential/dense/MatMulMatMul(sequential/bidirectional/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pw
sequential/activation/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P
sequential/dropout/IdentityIdentity(sequential/activation/Relu:activations:0*
T0*'
_output_shapes
:?????????P
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0­
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0―
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????P
NoOpNoOpJ^sequential/bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpI^sequential/bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpK^sequential/bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp-^sequential/bidirectional/backward_lstm/whileI^sequential/bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpH^sequential/bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOpJ^sequential/bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp,^sequential/bidirectional/forward_lstm/while(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 2
Isequential/bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpIsequential/bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2
Hsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpHsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2
Jsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpJsequential/bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2\
,sequential/bidirectional/backward_lstm/while,sequential/bidirectional/backward_lstm/while2
Hsequential/bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpHsequential/bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2
Gsequential/bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOpGsequential/bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2
Isequential/bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpIsequential/bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2Z
+sequential/bidirectional/forward_lstm/while+sequential/bidirectional/forward_lstm/while2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:` \
+
_output_shapes
:?????????P
-
_user_specified_namebidirectional_input
ς΄

H__inference_bidirectional_layer_call_and_return_conditional_losses_42315

inputsJ
7forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	ΘL
9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘG
8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	ΘK
8backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	ΘM
:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘH
9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’backward_lstm/while’/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’forward_lstm/whileH
forward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:j
 forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_sliceStridedSliceforward_lstm/Shape:output:0)forward_lstm/strided_slice/stack:output:0+forward_lstm/strided_slice/stack_1:output:0+forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros/packedPack#forward_lstm/strided_slice:output:0$forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zerosFill"forward_lstm/zeros/packed:output:0!forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2_
forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros_1/packedPack#forward_lstm/strided_slice:output:0&forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zeros_1Fill$forward_lstm/zeros_1/packed:output:0#forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2p
forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_lstm/transpose	Transposeinputs$forward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????^
forward_lstm/Shape_1Shapeforward_lstm/transpose:y:0*
T0*
_output_shapes
:l
"forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_slice_1StridedSliceforward_lstm/Shape_1:output:0+forward_lstm/strided_slice_1/stack:output:0-forward_lstm/strided_slice_1/stack_1:output:0-forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????Ϋ
forward_lstm/TensorArrayV2TensorListReserve1forward_lstm/TensorArrayV2/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
Bforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
4forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm/transpose:y:0Kforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?l
"forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ͺ
forward_lstm/strided_slice_2StridedSliceforward_lstm/transpose:y:0+forward_lstm/strided_slice_2/stack:output:0-forward_lstm/strided_slice_2/stack_1:output:0-forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask§
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp7forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0»
forward_lstm/lstm_cell_1/MatMulMatMul%forward_lstm/strided_slice_2:output:06forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ«
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0΅
!forward_lstm/lstm_cell_1/MatMul_1MatMulforward_lstm/zeros:output:08forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ°
forward_lstm/lstm_cell_1/addAddV2)forward_lstm/lstm_cell_1/MatMul:product:0+forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ₯
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ή
 forward_lstm/lstm_cell_1/BiasAddBiasAdd forward_lstm/lstm_cell_1/add:z:07forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θj
(forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/lstm_cell_1/splitSplit1forward_lstm/lstm_cell_1/split/split_dim:output:0)forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
 forward_lstm/lstm_cell_1/SigmoidSigmoid'forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid'forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/mulMul&forward_lstm/lstm_cell_1/Sigmoid_1:y:0forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/ReluRelu'forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2ͺ
forward_lstm/lstm_cell_1/mul_1Mul$forward_lstm/lstm_cell_1/Sigmoid:y:0+forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/add_1AddV2 forward_lstm/lstm_cell_1/mul:z:0"forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid'forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2}
forward_lstm/lstm_cell_1/Relu_1Relu"forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2?
forward_lstm/lstm_cell_1/mul_2Mul&forward_lstm/lstm_cell_1/Sigmoid_2:y:0-forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2{
*forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ί
forward_lstm/TensorArrayV2_1TensorListReserve3forward_lstm/TensorArrayV2_1/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?S
forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
forward_lstm/whileWhile(forward_lstm/while/loop_counter:output:0.forward_lstm/while/maximum_iterations:output:0forward_lstm/time:output:0%forward_lstm/TensorArrayV2_1:handle:0forward_lstm/zeros:output:0forward_lstm/zeros_1:output:0%forward_lstm/strided_slice_1:output:0Dforward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07forward_lstm_lstm_cell_1_matmul_readvariableop_resource9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
forward_lstm_while_body_42088*)
cond!R
forward_lstm_while_cond_42087*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
=forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ι
/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm/while:output:3Fforward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0u
"forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Θ
forward_lstm/strided_slice_3StridedSlice8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+forward_lstm/strided_slice_3/stack:output:0-forward_lstm/strided_slice_3/stack_1:output:0-forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maskr
forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
forward_lstm/transpose_1	Transpose8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&forward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2h
forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    I
backward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:k
!backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_lstm/strided_sliceStridedSlicebackward_lstm/Shape:output:0*backward_lstm/strided_slice/stack:output:0,backward_lstm/strided_slice/stack_1:output:0,backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm/zeros/packedPack$backward_lstm/strided_slice:output:0%backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zerosFill#backward_lstm/zeros/packed:output:0"backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2`
backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2‘
backward_lstm/zeros_1/packedPack$backward_lstm/strided_slice:output:0'backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:`
backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zeros_1Fill%backward_lstm/zeros_1/packed:output:0$backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2q
backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_lstm/transpose	Transposeinputs%backward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????`
backward_lstm/Shape_1Shapebackward_lstm/transpose:y:0*
T0*
_output_shapes
:m
#backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:‘
backward_lstm/strided_slice_1StridedSlicebackward_lstm/Shape_1:output:0,backward_lstm/strided_slice_1/stack:output:0.backward_lstm/strided_slice_1/stack_1:output:0.backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ή
backward_lstm/TensorArrayV2TensorListReserve2backward_lstm/TensorArrayV2/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?f
backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
backward_lstm/ReverseV2	ReverseV2backward_lstm/transpose:y:0%backward_lstm/ReverseV2/axis:output:0*
T0*+
_output_shapes
:P?????????
Cbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
5backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor backward_lstm/ReverseV2:output:0Lbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?m
#backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:―
backward_lstm/strided_slice_2StridedSlicebackward_lstm/transpose:y:0,backward_lstm/strided_slice_2/stack:output:0.backward_lstm/strided_slice_2/stack_1:output:0.backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask©
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp8backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0Ύ
 backward_lstm/lstm_cell_2/MatMulMatMul&backward_lstm/strided_slice_2:output:07backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ­
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0Έ
"backward_lstm/lstm_cell_2/MatMul_1MatMulbackward_lstm/zeros:output:09backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ³
backward_lstm/lstm_cell_2/addAddV2*backward_lstm/lstm_cell_2/MatMul:product:0,backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ§
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ό
!backward_lstm/lstm_cell_2/BiasAddBiasAdd!backward_lstm/lstm_cell_2/add:z:08backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θk
)backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/lstm_cell_2/splitSplit2backward_lstm/lstm_cell_2/split/split_dim:output:0*backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
!backward_lstm/lstm_cell_2/SigmoidSigmoid(backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_1Sigmoid(backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/mulMul'backward_lstm/lstm_cell_2/Sigmoid_1:y:0backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/ReluRelu(backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2­
backward_lstm/lstm_cell_2/mul_1Mul%backward_lstm/lstm_cell_2/Sigmoid:y:0,backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2’
backward_lstm/lstm_cell_2/add_1AddV2!backward_lstm/lstm_cell_2/mul:z:0#backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_2Sigmoid(backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
 backward_lstm/lstm_cell_2/Relu_1Relu#backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2±
backward_lstm/lstm_cell_2/mul_2Mul'backward_lstm/lstm_cell_2/Sigmoid_2:y:0.backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2|
+backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   β
backward_lstm/TensorArrayV2_1TensorListReserve4backward_lstm/TensorArrayV2_1/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?T
backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Α
backward_lstm/whileWhile)backward_lstm/while/loop_counter:output:0/backward_lstm/while/maximum_iterations:output:0backward_lstm/time:output:0&backward_lstm/TensorArrayV2_1:handle:0backward_lstm/zeros:output:0backward_lstm/zeros_1:output:0&backward_lstm/strided_slice_1:output:0Ebackward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08backward_lstm_lstm_cell_2_matmul_readvariableop_resource:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
backward_lstm_while_body_42229**
cond"R 
backward_lstm_while_cond_42228*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
>backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   μ
0backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm/while:output:3Gbackward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0v
#backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ν
backward_lstm/strided_slice_3StridedSlice9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0,backward_lstm/strided_slice_3/stack:output:0.backward_lstm/strided_slice_3/stack_1:output:0.backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_masks
backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ΐ
backward_lstm/transpose_1	Transpose9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0'backward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2i
backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concatConcatV2%forward_lstm/strided_slice_3:output:0&backward_lstm/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????d 
NoOpNoOp1^backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0^backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2^backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp^backward_lstm/while0^forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/^forward_lstm/lstm_cell_1/MatMul/ReadVariableOp1^forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp^forward_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : : : 2d
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2b
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2f
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2*
backward_lstm/whilebackward_lstm/while2b
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2`
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2d
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2(
forward_lstm/whileforward_lstm/while:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs

Β
forward_lstm_while_cond_415156
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_38
4forward_lstm_while_less_forward_lstm_strided_slice_1M
Iforward_lstm_while_forward_lstm_while_cond_41515___redundant_placeholder0M
Iforward_lstm_while_forward_lstm_while_cond_41515___redundant_placeholder1M
Iforward_lstm_while_forward_lstm_while_cond_41515___redundant_placeholder2M
Iforward_lstm_while_forward_lstm_while_cond_41515___redundant_placeholder3
forward_lstm_while_identity

forward_lstm/while/LessLessforward_lstm_while_placeholder4forward_lstm_while_less_forward_lstm_strided_slice_1*
T0*
_output_shapes
: e
forward_lstm/while/IdentityIdentityforward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
ΐJ

G__inference_forward_lstm_layer_call_and_return_conditional_losses_42863

inputs=
*lstm_cell_1_matmul_readvariableop_resource:	Θ?
,lstm_cell_1_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_1_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_1/BiasAdd/ReadVariableOp’!lstm_cell_1/MatMul/ReadVariableOp’#lstm_cell_1/MatMul_1/ReadVariableOp’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????ΰ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42779*
condR
while_cond_42778*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
H

backward_lstm_while_body_397128
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_37
3backward_lstm_while_backward_lstm_strided_slice_1_0s
obackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0S
@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘU
Bbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘP
Abackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ 
backward_lstm_while_identity"
backward_lstm_while_identity_1"
backward_lstm_while_identity_2"
backward_lstm_while_identity_3"
backward_lstm_while_identity_4"
backward_lstm_while_identity_55
1backward_lstm_while_backward_lstm_strided_slice_1q
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorQ
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	ΘS
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘN
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp
Ebackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   μ
7backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0backward_lstm_while_placeholderNbackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0·
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0β
&backward_lstm/while/lstm_cell_2/MatMulMatMul>backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ»
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ι
(backward_lstm/while/lstm_cell_2/MatMul_1MatMul!backward_lstm_while_placeholder_2?backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΕ
#backward_lstm/while/lstm_cell_2/addAddV20backward_lstm/while/lstm_cell_2/MatMul:product:02backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ΅
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Ξ
'backward_lstm/while/lstm_cell_2/BiasAddBiasAdd'backward_lstm/while/lstm_cell_2/add:z:0>backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θq
/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%backward_lstm/while/lstm_cell_2/splitSplit8backward_lstm/while/lstm_cell_2/split/split_dim:output:00backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
'backward_lstm/while/lstm_cell_2/SigmoidSigmoid.backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_1Sigmoid.backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2?
#backward_lstm/while/lstm_cell_2/mulMul-backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0!backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
$backward_lstm/while/lstm_cell_2/ReluRelu.backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2Ώ
%backward_lstm/while/lstm_cell_2/mul_1Mul+backward_lstm/while/lstm_cell_2/Sigmoid:y:02backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2΄
%backward_lstm/while/lstm_cell_2/add_1AddV2'backward_lstm/while/lstm_cell_2/mul:z:0)backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_2Sigmoid.backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
&backward_lstm/while/lstm_cell_2/Relu_1Relu)backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2Γ
%backward_lstm/while/lstm_cell_2/mul_2Mul-backward_lstm/while/lstm_cell_2/Sigmoid_2:y:04backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ό
8backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!backward_lstm_while_placeholder_1backward_lstm_while_placeholder)backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?[
backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/addAddV2backward_lstm_while_placeholder"backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: ]
backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/add_1AddV24backward_lstm_while_backward_lstm_while_loop_counter$backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_lstm/while/IdentityIdentitybackward_lstm/while/add_1:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: ’
backward_lstm/while/Identity_1Identity:backward_lstm_while_backward_lstm_while_maximum_iterations^backward_lstm/while/NoOp*
T0*
_output_shapes
: 
backward_lstm/while/Identity_2Identitybackward_lstm/while/add:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: Γ
backward_lstm/while/Identity_3IdentityHbackward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?’
backward_lstm/while/Identity_4Identity)backward_lstm/while/lstm_cell_2/mul_2:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2’
backward_lstm/while/Identity_5Identity)backward_lstm/while/lstm_cell_2/add_1:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
backward_lstm/while/NoOpNoOp7^backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6^backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp8^backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1backward_lstm_while_backward_lstm_strided_slice_13backward_lstm_while_backward_lstm_strided_slice_1_0"E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0"I
backward_lstm_while_identity_1'backward_lstm/while/Identity_1:output:0"I
backward_lstm_while_identity_2'backward_lstm/while/Identity_2:output:0"I
backward_lstm_while_identity_3'backward_lstm/while/Identity_3:output:0"I
backward_lstm_while_identity_4'backward_lstm/while/Identity_4:output:0"I
backward_lstm_while_identity_5'backward_lstm/while/Identity_5:output:0"
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resourceBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"ΰ
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2p
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2n
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2r
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ι
a
E__inference_activation_layer_call_and_return_conditional_losses_42344

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????PZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs

»
,__inference_forward_lstm_layer_call_fn_42412
inputs_0
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_forward_lstm_layer_call_and_return_conditional_losses_38447o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
£
Ή
,__inference_forward_lstm_layer_call_fn_42423

inputs
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity’StatefulPartitionedCallι
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_forward_lstm_layer_call_and_return_conditional_losses_38959o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
γ7
Ζ
while_body_43401
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_2_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_2/BiasAdd/ReadVariableOp’'while/lstm_cell_2/MatMul/ReadVariableOp’)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????―
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
©L

H__inference_backward_lstm_layer_call_and_return_conditional_losses_43630

inputs=
*lstm_cell_2_matmul_readvariableop_resource:	Θ?
,lstm_cell_2_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_2/BiasAdd/ReadVariableOp’!lstm_cell_2/MatMul/ReadVariableOp’#lstm_cell_2/MatMul_1/ReadVariableOp’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????ε
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_43546*
condR
while_cond_43545*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ϋ
έ
E__inference_sequential_layer_call_and_return_conditional_losses_40413
bidirectional_input&
bidirectional_40387:	Θ&
bidirectional_40389:	2Θ"
bidirectional_40391:	Θ&
bidirectional_40393:	Θ&
bidirectional_40395:	2Θ"
bidirectional_40397:	Θ
dense_40400:dP
dense_40402:P
dense_1_40407:PP
dense_1_40409:P
identity’%bidirectional/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dropout/StatefulPartitionedCallκ
%bidirectional/StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputbidirectional_40387bidirectional_40389bidirectional_40391bidirectional_40393bidirectional_40395bidirectional_40397*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_40236
dense/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_40400dense_40402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_39822Ϋ
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_39833β
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_39912
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_40407dense_1_40409*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_39852w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp&^bidirectional/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:` \
+
_output_shapes
:?????????P
-
_user_specified_namebidirectional_input
G
ζ
forward_lstm_while_body_395716
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_35
1forward_lstm_while_forward_lstm_strided_slice_1_0q
mforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0R
?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘT
Aforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘO
@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
forward_lstm_while_identity!
forward_lstm_while_identity_1!
forward_lstm_while_identity_2!
forward_lstm_while_identity_3!
forward_lstm_while_identity_4!
forward_lstm_while_identity_53
/forward_lstm_while_forward_lstm_strided_slice_1o
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensorP
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	ΘR
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘM
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp
Dforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   η
6forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0forward_lstm_while_placeholderMforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0΅
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0ί
%forward_lstm/while/lstm_cell_1/MatMulMatMul=forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΉ
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ζ
'forward_lstm/while/lstm_cell_1/MatMul_1MatMul forward_lstm_while_placeholder_2>forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΒ
"forward_lstm/while/lstm_cell_1/addAddV2/forward_lstm/while/lstm_cell_1/MatMul:product:01forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ³
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Λ
&forward_lstm/while/lstm_cell_1/BiasAddBiasAdd&forward_lstm/while/lstm_cell_1/add:z:0=forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θp
.forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$forward_lstm/while/lstm_cell_1/splitSplit7forward_lstm/while/lstm_cell_1/split/split_dim:output:0/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
&forward_lstm/while/lstm_cell_1/SigmoidSigmoid-forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_1Sigmoid-forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2«
"forward_lstm/while/lstm_cell_1/mulMul,forward_lstm/while/lstm_cell_1/Sigmoid_1:y:0 forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
#forward_lstm/while/lstm_cell_1/ReluRelu-forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2Ό
$forward_lstm/while/lstm_cell_1/mul_1Mul*forward_lstm/while/lstm_cell_1/Sigmoid:y:01forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2±
$forward_lstm/while/lstm_cell_1/add_1AddV2&forward_lstm/while/lstm_cell_1/mul:z:0(forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_2Sigmoid-forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
%forward_lstm/while/lstm_cell_1/Relu_1Relu(forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2ΐ
$forward_lstm/while/lstm_cell_1/mul_2Mul,forward_lstm/while/lstm_cell_1/Sigmoid_2:y:03forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ψ
7forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem forward_lstm_while_placeholder_1forward_lstm_while_placeholder(forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?Z
forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/addAddV2forward_lstm_while_placeholder!forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: \
forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/add_1AddV22forward_lstm_while_forward_lstm_while_loop_counter#forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_lstm/while/IdentityIdentityforward_lstm/while/add_1:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_1Identity8forward_lstm_while_forward_lstm_while_maximum_iterations^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_2Identityforward_lstm/while/add:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: ΐ
forward_lstm/while/Identity_3IdentityGforward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?
forward_lstm/while/Identity_4Identity(forward_lstm/while/lstm_cell_1/mul_2:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/Identity_5Identity(forward_lstm/while/lstm_cell_1/add_1:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/NoOpNoOp6^forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5^forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp7^forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/forward_lstm_while_forward_lstm_strided_slice_11forward_lstm_while_forward_lstm_strided_slice_1_0"C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0"G
forward_lstm_while_identity_1&forward_lstm/while/Identity_1:output:0"G
forward_lstm_while_identity_2&forward_lstm/while/Identity_2:output:0"G
forward_lstm_while_identity_3&forward_lstm/while/Identity_3:output:0"G
forward_lstm_while_identity_4&forward_lstm/while/Identity_4:output:0"G
forward_lstm_while_identity_5&forward_lstm/while/Identity_5:output:0"
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0"
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"ά
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensormforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2n
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2l
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2p
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
©L

H__inference_backward_lstm_layer_call_and_return_conditional_losses_39111

inputs=
*lstm_cell_2_matmul_readvariableop_resource:	Θ?
,lstm_cell_2_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_2/BiasAdd/ReadVariableOp’!lstm_cell_2/MatMul/ReadVariableOp’#lstm_cell_2/MatMul_1/ReadVariableOp’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????ε
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39027*
condR
while_cond_39026*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
G
ζ
forward_lstm_while_body_400096
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_35
1forward_lstm_while_forward_lstm_strided_slice_1_0q
mforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0R
?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘT
Aforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘO
@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
forward_lstm_while_identity!
forward_lstm_while_identity_1!
forward_lstm_while_identity_2!
forward_lstm_while_identity_3!
forward_lstm_while_identity_4!
forward_lstm_while_identity_53
/forward_lstm_while_forward_lstm_strided_slice_1o
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensorP
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	ΘR
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘM
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp
Dforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   η
6forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0forward_lstm_while_placeholderMforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0΅
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0ί
%forward_lstm/while/lstm_cell_1/MatMulMatMul=forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΉ
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ζ
'forward_lstm/while/lstm_cell_1/MatMul_1MatMul forward_lstm_while_placeholder_2>forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΒ
"forward_lstm/while/lstm_cell_1/addAddV2/forward_lstm/while/lstm_cell_1/MatMul:product:01forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ³
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Λ
&forward_lstm/while/lstm_cell_1/BiasAddBiasAdd&forward_lstm/while/lstm_cell_1/add:z:0=forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θp
.forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$forward_lstm/while/lstm_cell_1/splitSplit7forward_lstm/while/lstm_cell_1/split/split_dim:output:0/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
&forward_lstm/while/lstm_cell_1/SigmoidSigmoid-forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_1Sigmoid-forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2«
"forward_lstm/while/lstm_cell_1/mulMul,forward_lstm/while/lstm_cell_1/Sigmoid_1:y:0 forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
#forward_lstm/while/lstm_cell_1/ReluRelu-forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2Ό
$forward_lstm/while/lstm_cell_1/mul_1Mul*forward_lstm/while/lstm_cell_1/Sigmoid:y:01forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2±
$forward_lstm/while/lstm_cell_1/add_1AddV2&forward_lstm/while/lstm_cell_1/mul:z:0(forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_2Sigmoid-forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
%forward_lstm/while/lstm_cell_1/Relu_1Relu(forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2ΐ
$forward_lstm/while/lstm_cell_1/mul_2Mul,forward_lstm/while/lstm_cell_1/Sigmoid_2:y:03forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ψ
7forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem forward_lstm_while_placeholder_1forward_lstm_while_placeholder(forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?Z
forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/addAddV2forward_lstm_while_placeholder!forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: \
forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/add_1AddV22forward_lstm_while_forward_lstm_while_loop_counter#forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_lstm/while/IdentityIdentityforward_lstm/while/add_1:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_1Identity8forward_lstm_while_forward_lstm_while_maximum_iterations^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_2Identityforward_lstm/while/add:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: ΐ
forward_lstm/while/Identity_3IdentityGforward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?
forward_lstm/while/Identity_4Identity(forward_lstm/while/lstm_cell_1/mul_2:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/Identity_5Identity(forward_lstm/while/lstm_cell_1/add_1:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/NoOpNoOp6^forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5^forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp7^forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/forward_lstm_while_forward_lstm_strided_slice_11forward_lstm_while_forward_lstm_strided_slice_1_0"C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0"G
forward_lstm_while_identity_1&forward_lstm/while/Identity_1:output:0"G
forward_lstm_while_identity_2&forward_lstm/while/Identity_2:output:0"G
forward_lstm_while_identity_3&forward_lstm/while/Identity_3:output:0"G
forward_lstm_while_identity_4&forward_lstm/while/Identity_4:output:0"G
forward_lstm_while_identity_5&forward_lstm/while/Identity_5:output:0"
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0"
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"ά
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensormforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2n
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2l
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2p
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
	

-__inference_bidirectional_layer_call_fn_41171

inputs
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_40236o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
γ7
Ζ
while_body_39376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_1_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_1/BiasAdd/ReadVariableOp’'while/lstm_cell_1/MatMul/ReadVariableOp’)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????―
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
G
ζ
forward_lstm_while_body_415166
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_35
1forward_lstm_while_forward_lstm_strided_slice_1_0q
mforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0R
?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘT
Aforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘO
@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
forward_lstm_while_identity!
forward_lstm_while_identity_1!
forward_lstm_while_identity_2!
forward_lstm_while_identity_3!
forward_lstm_while_identity_4!
forward_lstm_while_identity_53
/forward_lstm_while_forward_lstm_strided_slice_1o
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensorP
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	ΘR
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘM
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp
Dforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????π
6forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0forward_lstm_while_placeholderMforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0΅
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0ί
%forward_lstm/while/lstm_cell_1/MatMulMatMul=forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΉ
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ζ
'forward_lstm/while/lstm_cell_1/MatMul_1MatMul forward_lstm_while_placeholder_2>forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΒ
"forward_lstm/while/lstm_cell_1/addAddV2/forward_lstm/while/lstm_cell_1/MatMul:product:01forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ³
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Λ
&forward_lstm/while/lstm_cell_1/BiasAddBiasAdd&forward_lstm/while/lstm_cell_1/add:z:0=forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θp
.forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$forward_lstm/while/lstm_cell_1/splitSplit7forward_lstm/while/lstm_cell_1/split/split_dim:output:0/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
&forward_lstm/while/lstm_cell_1/SigmoidSigmoid-forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_1Sigmoid-forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2«
"forward_lstm/while/lstm_cell_1/mulMul,forward_lstm/while/lstm_cell_1/Sigmoid_1:y:0 forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
#forward_lstm/while/lstm_cell_1/ReluRelu-forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2Ό
$forward_lstm/while/lstm_cell_1/mul_1Mul*forward_lstm/while/lstm_cell_1/Sigmoid:y:01forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2±
$forward_lstm/while/lstm_cell_1/add_1AddV2&forward_lstm/while/lstm_cell_1/mul:z:0(forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_2Sigmoid-forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
%forward_lstm/while/lstm_cell_1/Relu_1Relu(forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2ΐ
$forward_lstm/while/lstm_cell_1/mul_2Mul,forward_lstm/while/lstm_cell_1/Sigmoid_2:y:03forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ψ
7forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem forward_lstm_while_placeholder_1forward_lstm_while_placeholder(forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?Z
forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/addAddV2forward_lstm_while_placeholder!forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: \
forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/add_1AddV22forward_lstm_while_forward_lstm_while_loop_counter#forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_lstm/while/IdentityIdentityforward_lstm/while/add_1:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_1Identity8forward_lstm_while_forward_lstm_while_maximum_iterations^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_2Identityforward_lstm/while/add:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: ΐ
forward_lstm/while/Identity_3IdentityGforward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?
forward_lstm/while/Identity_4Identity(forward_lstm/while/lstm_cell_1/mul_2:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/Identity_5Identity(forward_lstm/while/lstm_cell_1/add_1:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/NoOpNoOp6^forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5^forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp7^forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/forward_lstm_while_forward_lstm_strided_slice_11forward_lstm_while_forward_lstm_strided_slice_1_0"C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0"G
forward_lstm_while_identity_1&forward_lstm/while/Identity_1:output:0"G
forward_lstm_while_identity_2&forward_lstm/while/Identity_2:output:0"G
forward_lstm_while_identity_3&forward_lstm/while/Identity_3:output:0"G
forward_lstm_while_identity_4&forward_lstm/while/Identity_4:output:0"G
forward_lstm_while_identity_5&forward_lstm/while/Identity_5:output:0"
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0"
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"ά
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensormforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2n
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2l
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2p
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Σ

F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_38319

inputs

states
states_11
matmul_readvariableop_resource:	Θ3
 matmul_1_readvariableop_resource:	2Θ.
biasadd_readvariableop_resource:	Θ
identity

identity_1

identity_2’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ά
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
Ύ

'__inference_dense_1_layer_call_fn_42380

inputs
unknown:PP
	unknown_0:P
identity’StatefulPartitionedCallΧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_39852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
Ϊ7
Ζ
while_body_43256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_2_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_2/BiasAdd/ReadVariableOp’'while/lstm_cell_2/MatMul/ReadVariableOp’)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
G
ζ
forward_lstm_while_body_412306
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_35
1forward_lstm_while_forward_lstm_strided_slice_1_0q
mforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0R
?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘT
Aforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘO
@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
forward_lstm_while_identity!
forward_lstm_while_identity_1!
forward_lstm_while_identity_2!
forward_lstm_while_identity_3!
forward_lstm_while_identity_4!
forward_lstm_while_identity_53
/forward_lstm_while_forward_lstm_strided_slice_1o
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensorP
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	ΘR
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘM
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp
Dforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????π
6forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0forward_lstm_while_placeholderMforward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0΅
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0ί
%forward_lstm/while/lstm_cell_1/MatMulMatMul=forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΉ
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ζ
'forward_lstm/while/lstm_cell_1/MatMul_1MatMul forward_lstm_while_placeholder_2>forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΒ
"forward_lstm/while/lstm_cell_1/addAddV2/forward_lstm/while/lstm_cell_1/MatMul:product:01forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ³
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Λ
&forward_lstm/while/lstm_cell_1/BiasAddBiasAdd&forward_lstm/while/lstm_cell_1/add:z:0=forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θp
.forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$forward_lstm/while/lstm_cell_1/splitSplit7forward_lstm/while/lstm_cell_1/split/split_dim:output:0/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
&forward_lstm/while/lstm_cell_1/SigmoidSigmoid-forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_1Sigmoid-forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2«
"forward_lstm/while/lstm_cell_1/mulMul,forward_lstm/while/lstm_cell_1/Sigmoid_1:y:0 forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
#forward_lstm/while/lstm_cell_1/ReluRelu-forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2Ό
$forward_lstm/while/lstm_cell_1/mul_1Mul*forward_lstm/while/lstm_cell_1/Sigmoid:y:01forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2±
$forward_lstm/while/lstm_cell_1/add_1AddV2&forward_lstm/while/lstm_cell_1/mul:z:0(forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
(forward_lstm/while/lstm_cell_1/Sigmoid_2Sigmoid-forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
%forward_lstm/while/lstm_cell_1/Relu_1Relu(forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2ΐ
$forward_lstm/while/lstm_cell_1/mul_2Mul,forward_lstm/while/lstm_cell_1/Sigmoid_2:y:03forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ψ
7forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem forward_lstm_while_placeholder_1forward_lstm_while_placeholder(forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?Z
forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/addAddV2forward_lstm_while_placeholder!forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: \
forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/while/add_1AddV22forward_lstm_while_forward_lstm_while_loop_counter#forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_lstm/while/IdentityIdentityforward_lstm/while/add_1:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_1Identity8forward_lstm_while_forward_lstm_while_maximum_iterations^forward_lstm/while/NoOp*
T0*
_output_shapes
: 
forward_lstm/while/Identity_2Identityforward_lstm/while/add:z:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: ΐ
forward_lstm/while/Identity_3IdentityGforward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?
forward_lstm/while/Identity_4Identity(forward_lstm/while/lstm_cell_1/mul_2:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/Identity_5Identity(forward_lstm/while/lstm_cell_1/add_1:z:0^forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
forward_lstm/while/NoOpNoOp6^forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5^forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp7^forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/forward_lstm_while_forward_lstm_strided_slice_11forward_lstm_while_forward_lstm_strided_slice_1_0"C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0"G
forward_lstm_while_identity_1&forward_lstm/while/Identity_1:output:0"G
forward_lstm_while_identity_2&forward_lstm/while/Identity_2:output:0"G
forward_lstm_while_identity_3&forward_lstm/while/Identity_3:output:0"G
forward_lstm_while_identity_4&forward_lstm/while/Identity_4:output:0"G
forward_lstm_while_identity_5&forward_lstm/while/Identity_5:output:0"
>forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource@forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0"
?forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceAforward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
=forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource?forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"ά
kforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensormforward_lstm_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2n
5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp5forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2l
4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp4forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2p
6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp6forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ά

H__inference_bidirectional_layer_call_and_return_conditional_losses_41743
inputs_0J
7forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	ΘL
9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘG
8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	ΘK
8backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	ΘM
:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘH
9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’backward_lstm/while’/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’forward_lstm/whileJ
forward_lstm/ShapeShapeinputs_0*
T0*
_output_shapes
:j
 forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_sliceStridedSliceforward_lstm/Shape:output:0)forward_lstm/strided_slice/stack:output:0+forward_lstm/strided_slice/stack_1:output:0+forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros/packedPack#forward_lstm/strided_slice:output:0$forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zerosFill"forward_lstm/zeros/packed:output:0!forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2_
forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros_1/packedPack#forward_lstm/strided_slice:output:0&forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zeros_1Fill$forward_lstm/zeros_1/packed:output:0#forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2p
forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_lstm/transpose	Transposeinputs_0$forward_lstm/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????^
forward_lstm/Shape_1Shapeforward_lstm/transpose:y:0*
T0*
_output_shapes
:l
"forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_slice_1StridedSliceforward_lstm/Shape_1:output:0+forward_lstm/strided_slice_1/stack:output:0-forward_lstm/strided_slice_1/stack_1:output:0-forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????Ϋ
forward_lstm/TensorArrayV2TensorListReserve1forward_lstm/TensorArrayV2/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
Bforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????
4forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm/transpose:y:0Kforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?l
"forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
forward_lstm/strided_slice_2StridedSliceforward_lstm/transpose:y:0+forward_lstm/strided_slice_2/stack:output:0-forward_lstm/strided_slice_2/stack_1:output:0-forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask§
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp7forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0»
forward_lstm/lstm_cell_1/MatMulMatMul%forward_lstm/strided_slice_2:output:06forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ«
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0΅
!forward_lstm/lstm_cell_1/MatMul_1MatMulforward_lstm/zeros:output:08forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ°
forward_lstm/lstm_cell_1/addAddV2)forward_lstm/lstm_cell_1/MatMul:product:0+forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ₯
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ή
 forward_lstm/lstm_cell_1/BiasAddBiasAdd forward_lstm/lstm_cell_1/add:z:07forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θj
(forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/lstm_cell_1/splitSplit1forward_lstm/lstm_cell_1/split/split_dim:output:0)forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
 forward_lstm/lstm_cell_1/SigmoidSigmoid'forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid'forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/mulMul&forward_lstm/lstm_cell_1/Sigmoid_1:y:0forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/ReluRelu'forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2ͺ
forward_lstm/lstm_cell_1/mul_1Mul$forward_lstm/lstm_cell_1/Sigmoid:y:0+forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/add_1AddV2 forward_lstm/lstm_cell_1/mul:z:0"forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid'forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2}
forward_lstm/lstm_cell_1/Relu_1Relu"forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2?
forward_lstm/lstm_cell_1/mul_2Mul&forward_lstm/lstm_cell_1/Sigmoid_2:y:0-forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2{
*forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ί
forward_lstm/TensorArrayV2_1TensorListReserve3forward_lstm/TensorArrayV2_1/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?S
forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
forward_lstm/whileWhile(forward_lstm/while/loop_counter:output:0.forward_lstm/while/maximum_iterations:output:0forward_lstm/time:output:0%forward_lstm/TensorArrayV2_1:handle:0forward_lstm/zeros:output:0forward_lstm/zeros_1:output:0%forward_lstm/strided_slice_1:output:0Dforward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07forward_lstm_lstm_cell_1_matmul_readvariableop_resource9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
forward_lstm_while_body_41516*)
cond!R
forward_lstm_while_cond_41515*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
=forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ς
/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm/while:output:3Fforward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0u
"forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Θ
forward_lstm/strided_slice_3StridedSlice8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+forward_lstm/strided_slice_3/stack:output:0-forward_lstm/strided_slice_3/stack_1:output:0-forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maskr
forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ζ
forward_lstm/transpose_1	Transpose8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&forward_lstm/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2h
forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    K
backward_lstm/ShapeShapeinputs_0*
T0*
_output_shapes
:k
!backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_lstm/strided_sliceStridedSlicebackward_lstm/Shape:output:0*backward_lstm/strided_slice/stack:output:0,backward_lstm/strided_slice/stack_1:output:0,backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm/zeros/packedPack$backward_lstm/strided_slice:output:0%backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zerosFill#backward_lstm/zeros/packed:output:0"backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2`
backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2‘
backward_lstm/zeros_1/packedPack$backward_lstm/strided_slice:output:0'backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:`
backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zeros_1Fill%backward_lstm/zeros_1/packed:output:0$backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2q
backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_lstm/transpose	Transposeinputs_0%backward_lstm/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
backward_lstm/Shape_1Shapebackward_lstm/transpose:y:0*
T0*
_output_shapes
:m
#backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:‘
backward_lstm/strided_slice_1StridedSlicebackward_lstm/Shape_1:output:0,backward_lstm/strided_slice_1/stack:output:0.backward_lstm/strided_slice_1/stack_1:output:0.backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ή
backward_lstm/TensorArrayV2TensorListReserve2backward_lstm/TensorArrayV2/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?f
backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: °
backward_lstm/ReverseV2	ReverseV2backward_lstm/transpose:y:0%backward_lstm/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????
Cbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????
5backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor backward_lstm/ReverseV2:output:0Lbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?m
#backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Έ
backward_lstm/strided_slice_2StridedSlicebackward_lstm/transpose:y:0,backward_lstm/strided_slice_2/stack:output:0.backward_lstm/strided_slice_2/stack_1:output:0.backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask©
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp8backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0Ύ
 backward_lstm/lstm_cell_2/MatMulMatMul&backward_lstm/strided_slice_2:output:07backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ­
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0Έ
"backward_lstm/lstm_cell_2/MatMul_1MatMulbackward_lstm/zeros:output:09backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ³
backward_lstm/lstm_cell_2/addAddV2*backward_lstm/lstm_cell_2/MatMul:product:0,backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ§
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ό
!backward_lstm/lstm_cell_2/BiasAddBiasAdd!backward_lstm/lstm_cell_2/add:z:08backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θk
)backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/lstm_cell_2/splitSplit2backward_lstm/lstm_cell_2/split/split_dim:output:0*backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
!backward_lstm/lstm_cell_2/SigmoidSigmoid(backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_1Sigmoid(backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/mulMul'backward_lstm/lstm_cell_2/Sigmoid_1:y:0backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/ReluRelu(backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2­
backward_lstm/lstm_cell_2/mul_1Mul%backward_lstm/lstm_cell_2/Sigmoid:y:0,backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2’
backward_lstm/lstm_cell_2/add_1AddV2!backward_lstm/lstm_cell_2/mul:z:0#backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_2Sigmoid(backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
 backward_lstm/lstm_cell_2/Relu_1Relu#backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2±
backward_lstm/lstm_cell_2/mul_2Mul'backward_lstm/lstm_cell_2/Sigmoid_2:y:0.backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2|
+backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   β
backward_lstm/TensorArrayV2_1TensorListReserve4backward_lstm/TensorArrayV2_1/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?T
backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Α
backward_lstm/whileWhile)backward_lstm/while/loop_counter:output:0/backward_lstm/while/maximum_iterations:output:0backward_lstm/time:output:0&backward_lstm/TensorArrayV2_1:handle:0backward_lstm/zeros:output:0backward_lstm/zeros_1:output:0&backward_lstm/strided_slice_1:output:0Ebackward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08backward_lstm_lstm_cell_2_matmul_readvariableop_resource:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
backward_lstm_while_body_41657**
cond"R 
backward_lstm_while_cond_41656*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
>backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   υ
0backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm/while:output:3Gbackward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0v
#backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ν
backward_lstm/strided_slice_3StridedSlice9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0,backward_lstm/strided_slice_3/stack:output:0.backward_lstm/strided_slice_3/stack_1:output:0.backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_masks
backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ι
backward_lstm/transpose_1	Transpose9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0'backward_lstm/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2i
backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concatConcatV2%forward_lstm/strided_slice_3:output:0&backward_lstm/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????d 
NoOpNoOp1^backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0^backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2^backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp^backward_lstm/while0^forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/^forward_lstm/lstm_cell_1/MatMul/ReadVariableOp1^forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp^forward_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2d
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2b
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2f
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2*
backward_lstm/whilebackward_lstm/while2b
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2`
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2d
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2(
forward_lstm/whileforward_lstm/while:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
γ7
Ζ
while_body_39027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_2_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_2/BiasAdd/ReadVariableOp’'while/lstm_cell_2/MatMul/ReadVariableOp’)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????―
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
λ
`
'__inference_dropout_layer_call_fn_42354

inputs
identity’StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_39912o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
€J

G__inference_forward_lstm_layer_call_and_return_conditional_losses_42577
inputs_0=
*lstm_cell_1_matmul_readvariableop_resource:	Θ?
,lstm_cell_1_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_1_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_1/BiasAdd/ReadVariableOp’!lstm_cell_1/MatMul/ReadVariableOp’#lstm_cell_1/MatMul_1/ReadVariableOp’while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ΰ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42493*
condR
while_cond_42492*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
ΖW
ͺ
+bidirectional_forward_lstm_while_body_40528R
Nbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_loop_counterX
Tbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_maximum_iterations0
,bidirectional_forward_lstm_while_placeholder2
.bidirectional_forward_lstm_while_placeholder_12
.bidirectional_forward_lstm_while_placeholder_22
.bidirectional_forward_lstm_while_placeholder_3Q
Mbidirectional_forward_lstm_while_bidirectional_forward_lstm_strided_slice_1_0
bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0`
Mbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	Θb
Obidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2Θ]
Nbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ-
)bidirectional_forward_lstm_while_identity/
+bidirectional_forward_lstm_while_identity_1/
+bidirectional_forward_lstm_while_identity_2/
+bidirectional_forward_lstm_while_identity_3/
+bidirectional_forward_lstm_while_identity_4/
+bidirectional_forward_lstm_while_identity_5O
Kbidirectional_forward_lstm_while_bidirectional_forward_lstm_strided_slice_1
bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor^
Kbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	Θ`
Mbidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ[
Lbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’Cbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’Bbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’Dbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp£
Rbidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Dbidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0,bidirectional_forward_lstm_while_placeholder[bidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0Ρ
Bbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpMbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0
3bidirectional/forward_lstm/while/lstm_cell_1/MatMulMatMulKbidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Jbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΥ
Dbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpObidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0π
5bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1MatMul.bidirectional_forward_lstm_while_placeholder_2Lbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θμ
0bidirectional/forward_lstm/while/lstm_cell_1/addAddV2=bidirectional/forward_lstm/while/lstm_cell_1/MatMul:product:0?bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΟ
Cbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpNbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0υ
4bidirectional/forward_lstm/while/lstm_cell_1/BiasAddBiasAdd4bidirectional/forward_lstm/while/lstm_cell_1/add:z:0Kbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ~
<bidirectional/forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :½
2bidirectional/forward_lstm/while/lstm_cell_1/splitSplitEbidirectional/forward_lstm/while/lstm_cell_1/split/split_dim:output:0=bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split?
4bidirectional/forward_lstm/while/lstm_cell_1/SigmoidSigmoid;bidirectional/forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2°
6bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_1Sigmoid;bidirectional/forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2Υ
0bidirectional/forward_lstm/while/lstm_cell_1/mulMul:bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_1:y:0.bidirectional_forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2¨
1bidirectional/forward_lstm/while/lstm_cell_1/ReluRelu;bidirectional/forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2ζ
2bidirectional/forward_lstm/while/lstm_cell_1/mul_1Mul8bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid:y:0?bidirectional/forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2Ϋ
2bidirectional/forward_lstm/while/lstm_cell_1/add_1AddV24bidirectional/forward_lstm/while/lstm_cell_1/mul:z:06bidirectional/forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2°
6bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_2Sigmoid;bidirectional/forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2₯
3bidirectional/forward_lstm/while/lstm_cell_1/Relu_1Relu6bidirectional/forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2κ
2bidirectional/forward_lstm/while/lstm_cell_1/mul_2Mul:bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_2:y:0Abidirectional/forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2°
Ebidirectional/forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.bidirectional_forward_lstm_while_placeholder_1,bidirectional_forward_lstm_while_placeholder6bidirectional/forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?h
&bidirectional/forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :­
$bidirectional/forward_lstm/while/addAddV2,bidirectional_forward_lstm_while_placeholder/bidirectional/forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: j
(bidirectional/forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Σ
&bidirectional/forward_lstm/while/add_1AddV2Nbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_loop_counter1bidirectional/forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: ͺ
)bidirectional/forward_lstm/while/IdentityIdentity*bidirectional/forward_lstm/while/add_1:z:0&^bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: Φ
+bidirectional/forward_lstm/while/Identity_1IdentityTbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_maximum_iterations&^bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: ͺ
+bidirectional/forward_lstm/while/Identity_2Identity(bidirectional/forward_lstm/while/add:z:0&^bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: κ
+bidirectional/forward_lstm/while/Identity_3IdentityUbidirectional/forward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?Ι
+bidirectional/forward_lstm/while/Identity_4Identity6bidirectional/forward_lstm/while/lstm_cell_1/mul_2:z:0&^bidirectional/forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2Ι
+bidirectional/forward_lstm/while/Identity_5Identity6bidirectional/forward_lstm/while/lstm_cell_1/add_1:z:0&^bidirectional/forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2Ή
%bidirectional/forward_lstm/while/NoOpNoOpD^bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpC^bidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpE^bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Kbidirectional_forward_lstm_while_bidirectional_forward_lstm_strided_slice_1Mbidirectional_forward_lstm_while_bidirectional_forward_lstm_strided_slice_1_0"_
)bidirectional_forward_lstm_while_identity2bidirectional/forward_lstm/while/Identity:output:0"c
+bidirectional_forward_lstm_while_identity_14bidirectional/forward_lstm/while/Identity_1:output:0"c
+bidirectional_forward_lstm_while_identity_24bidirectional/forward_lstm/while/Identity_2:output:0"c
+bidirectional_forward_lstm_while_identity_34bidirectional/forward_lstm/while/Identity_3:output:0"c
+bidirectional_forward_lstm_while_identity_44bidirectional/forward_lstm/while/Identity_4:output:0"c
+bidirectional_forward_lstm_while_identity_54bidirectional/forward_lstm/while/Identity_5:output:0"
Lbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resourceNbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0" 
Mbidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceObidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
Kbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resourceMbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"
bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensorbidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2
Cbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpCbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2
Bbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpBbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2
Dbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpDbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
°
Ύ
while_cond_43400
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_43400___redundant_placeholder03
/while_while_cond_43400___redundant_placeholder13
/while_while_cond_43400___redundant_placeholder23
/while_while_cond_43400___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
ς΄

H__inference_bidirectional_layer_call_and_return_conditional_losses_39798

inputsJ
7forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	ΘL
9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘG
8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	ΘK
8backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	ΘM
:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘH
9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’backward_lstm/while’/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’forward_lstm/whileH
forward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:j
 forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_sliceStridedSliceforward_lstm/Shape:output:0)forward_lstm/strided_slice/stack:output:0+forward_lstm/strided_slice/stack_1:output:0+forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros/packedPack#forward_lstm/strided_slice:output:0$forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zerosFill"forward_lstm/zeros/packed:output:0!forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2_
forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros_1/packedPack#forward_lstm/strided_slice:output:0&forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zeros_1Fill$forward_lstm/zeros_1/packed:output:0#forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2p
forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_lstm/transpose	Transposeinputs$forward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????^
forward_lstm/Shape_1Shapeforward_lstm/transpose:y:0*
T0*
_output_shapes
:l
"forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_slice_1StridedSliceforward_lstm/Shape_1:output:0+forward_lstm/strided_slice_1/stack:output:0-forward_lstm/strided_slice_1/stack_1:output:0-forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????Ϋ
forward_lstm/TensorArrayV2TensorListReserve1forward_lstm/TensorArrayV2/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
Bforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
4forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm/transpose:y:0Kforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?l
"forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ͺ
forward_lstm/strided_slice_2StridedSliceforward_lstm/transpose:y:0+forward_lstm/strided_slice_2/stack:output:0-forward_lstm/strided_slice_2/stack_1:output:0-forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask§
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp7forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0»
forward_lstm/lstm_cell_1/MatMulMatMul%forward_lstm/strided_slice_2:output:06forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ«
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0΅
!forward_lstm/lstm_cell_1/MatMul_1MatMulforward_lstm/zeros:output:08forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ°
forward_lstm/lstm_cell_1/addAddV2)forward_lstm/lstm_cell_1/MatMul:product:0+forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ₯
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ή
 forward_lstm/lstm_cell_1/BiasAddBiasAdd forward_lstm/lstm_cell_1/add:z:07forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θj
(forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/lstm_cell_1/splitSplit1forward_lstm/lstm_cell_1/split/split_dim:output:0)forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
 forward_lstm/lstm_cell_1/SigmoidSigmoid'forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid'forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/mulMul&forward_lstm/lstm_cell_1/Sigmoid_1:y:0forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/ReluRelu'forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2ͺ
forward_lstm/lstm_cell_1/mul_1Mul$forward_lstm/lstm_cell_1/Sigmoid:y:0+forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/add_1AddV2 forward_lstm/lstm_cell_1/mul:z:0"forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid'forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2}
forward_lstm/lstm_cell_1/Relu_1Relu"forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2?
forward_lstm/lstm_cell_1/mul_2Mul&forward_lstm/lstm_cell_1/Sigmoid_2:y:0-forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2{
*forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ί
forward_lstm/TensorArrayV2_1TensorListReserve3forward_lstm/TensorArrayV2_1/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?S
forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
forward_lstm/whileWhile(forward_lstm/while/loop_counter:output:0.forward_lstm/while/maximum_iterations:output:0forward_lstm/time:output:0%forward_lstm/TensorArrayV2_1:handle:0forward_lstm/zeros:output:0forward_lstm/zeros_1:output:0%forward_lstm/strided_slice_1:output:0Dforward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07forward_lstm_lstm_cell_1_matmul_readvariableop_resource9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
forward_lstm_while_body_39571*)
cond!R
forward_lstm_while_cond_39570*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
=forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ι
/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm/while:output:3Fforward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0u
"forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Θ
forward_lstm/strided_slice_3StridedSlice8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+forward_lstm/strided_slice_3/stack:output:0-forward_lstm/strided_slice_3/stack_1:output:0-forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maskr
forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
forward_lstm/transpose_1	Transpose8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&forward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2h
forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    I
backward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:k
!backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_lstm/strided_sliceStridedSlicebackward_lstm/Shape:output:0*backward_lstm/strided_slice/stack:output:0,backward_lstm/strided_slice/stack_1:output:0,backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm/zeros/packedPack$backward_lstm/strided_slice:output:0%backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zerosFill#backward_lstm/zeros/packed:output:0"backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2`
backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2‘
backward_lstm/zeros_1/packedPack$backward_lstm/strided_slice:output:0'backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:`
backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zeros_1Fill%backward_lstm/zeros_1/packed:output:0$backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2q
backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_lstm/transpose	Transposeinputs%backward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????`
backward_lstm/Shape_1Shapebackward_lstm/transpose:y:0*
T0*
_output_shapes
:m
#backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:‘
backward_lstm/strided_slice_1StridedSlicebackward_lstm/Shape_1:output:0,backward_lstm/strided_slice_1/stack:output:0.backward_lstm/strided_slice_1/stack_1:output:0.backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ή
backward_lstm/TensorArrayV2TensorListReserve2backward_lstm/TensorArrayV2/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?f
backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
backward_lstm/ReverseV2	ReverseV2backward_lstm/transpose:y:0%backward_lstm/ReverseV2/axis:output:0*
T0*+
_output_shapes
:P?????????
Cbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
5backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor backward_lstm/ReverseV2:output:0Lbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?m
#backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:―
backward_lstm/strided_slice_2StridedSlicebackward_lstm/transpose:y:0,backward_lstm/strided_slice_2/stack:output:0.backward_lstm/strided_slice_2/stack_1:output:0.backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask©
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp8backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0Ύ
 backward_lstm/lstm_cell_2/MatMulMatMul&backward_lstm/strided_slice_2:output:07backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ­
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0Έ
"backward_lstm/lstm_cell_2/MatMul_1MatMulbackward_lstm/zeros:output:09backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ³
backward_lstm/lstm_cell_2/addAddV2*backward_lstm/lstm_cell_2/MatMul:product:0,backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ§
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ό
!backward_lstm/lstm_cell_2/BiasAddBiasAdd!backward_lstm/lstm_cell_2/add:z:08backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θk
)backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/lstm_cell_2/splitSplit2backward_lstm/lstm_cell_2/split/split_dim:output:0*backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
!backward_lstm/lstm_cell_2/SigmoidSigmoid(backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_1Sigmoid(backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/mulMul'backward_lstm/lstm_cell_2/Sigmoid_1:y:0backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/ReluRelu(backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2­
backward_lstm/lstm_cell_2/mul_1Mul%backward_lstm/lstm_cell_2/Sigmoid:y:0,backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2’
backward_lstm/lstm_cell_2/add_1AddV2!backward_lstm/lstm_cell_2/mul:z:0#backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_2Sigmoid(backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
 backward_lstm/lstm_cell_2/Relu_1Relu#backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2±
backward_lstm/lstm_cell_2/mul_2Mul'backward_lstm/lstm_cell_2/Sigmoid_2:y:0.backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2|
+backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   β
backward_lstm/TensorArrayV2_1TensorListReserve4backward_lstm/TensorArrayV2_1/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?T
backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Α
backward_lstm/whileWhile)backward_lstm/while/loop_counter:output:0/backward_lstm/while/maximum_iterations:output:0backward_lstm/time:output:0&backward_lstm/TensorArrayV2_1:handle:0backward_lstm/zeros:output:0backward_lstm/zeros_1:output:0&backward_lstm/strided_slice_1:output:0Ebackward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08backward_lstm_lstm_cell_2_matmul_readvariableop_resource:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
backward_lstm_while_body_39712**
cond"R 
backward_lstm_while_cond_39711*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
>backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   μ
0backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm/while:output:3Gbackward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0v
#backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ν
backward_lstm/strided_slice_3StridedSlice9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0,backward_lstm/strided_slice_3/stack:output:0.backward_lstm/strided_slice_3/stack_1:output:0.backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_masks
backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ΐ
backward_lstm/transpose_1	Transpose9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0'backward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2i
backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concatConcatV2%forward_lstm/strided_slice_3:output:0&backward_lstm/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????d 
NoOpNoOp1^backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0^backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2^backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp^backward_lstm/while0^forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/^forward_lstm/lstm_cell_1/MatMul/ReadVariableOp1^forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp^forward_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : : : 2d
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2b
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2f
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2*
backward_lstm/whilebackward_lstm/while2b
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2`
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2d
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2(
forward_lstm/whileforward_lstm/while:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_39912

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed2????[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????Po
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????Pi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????PY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs

F
*__inference_activation_layer_call_fn_42339

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_39833`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
₯
Ί
-__inference_backward_lstm_layer_call_fn_43039

inputs
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_39111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ΐJ

G__inference_forward_lstm_layer_call_and_return_conditional_losses_38959

inputs=
*lstm_cell_1_matmul_readvariableop_resource:	Θ?
,lstm_cell_1_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_1_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_1/BiasAdd/ReadVariableOp’!lstm_cell_1/MatMul/ReadVariableOp’#lstm_cell_1/MatMul_1/ReadVariableOp’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????ΰ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38875*
condR
while_cond_38874*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
₯
Ί
-__inference_backward_lstm_layer_call_fn_43050

inputs
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_39295o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Β
ξ
,bidirectional_backward_lstm_while_cond_40968T
Pbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_loop_counterZ
Vbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_maximum_iterations1
-bidirectional_backward_lstm_while_placeholder3
/bidirectional_backward_lstm_while_placeholder_13
/bidirectional_backward_lstm_while_placeholder_23
/bidirectional_backward_lstm_while_placeholder_3V
Rbidirectional_backward_lstm_while_less_bidirectional_backward_lstm_strided_slice_1k
gbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_cond_40968___redundant_placeholder0k
gbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_cond_40968___redundant_placeholder1k
gbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_cond_40968___redundant_placeholder2k
gbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_cond_40968___redundant_placeholder3.
*bidirectional_backward_lstm_while_identity
?
&bidirectional/backward_lstm/while/LessLess-bidirectional_backward_lstm_while_placeholderRbidirectional_backward_lstm_while_less_bidirectional_backward_lstm_strided_slice_1*
T0*
_output_shapes
: 
*bidirectional/backward_lstm/while/IdentityIdentity*bidirectional/backward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "a
*bidirectional_backward_lstm_while_identity3bidirectional/backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
°
Ύ
while_cond_39375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_39375___redundant_placeholder03
/while_while_cond_39375___redundant_placeholder13
/while_while_cond_39375___redundant_placeholder23
/while_while_cond_39375___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
ΖW
ͺ
+bidirectional_forward_lstm_while_body_40828R
Nbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_loop_counterX
Tbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_maximum_iterations0
,bidirectional_forward_lstm_while_placeholder2
.bidirectional_forward_lstm_while_placeholder_12
.bidirectional_forward_lstm_while_placeholder_22
.bidirectional_forward_lstm_while_placeholder_3Q
Mbidirectional_forward_lstm_while_bidirectional_forward_lstm_strided_slice_1_0
bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0`
Mbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	Θb
Obidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2Θ]
Nbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ-
)bidirectional_forward_lstm_while_identity/
+bidirectional_forward_lstm_while_identity_1/
+bidirectional_forward_lstm_while_identity_2/
+bidirectional_forward_lstm_while_identity_3/
+bidirectional_forward_lstm_while_identity_4/
+bidirectional_forward_lstm_while_identity_5O
Kbidirectional_forward_lstm_while_bidirectional_forward_lstm_strided_slice_1
bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor^
Kbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	Θ`
Mbidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ[
Lbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’Cbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’Bbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’Dbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp£
Rbidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Dbidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0,bidirectional_forward_lstm_while_placeholder[bidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0Ρ
Bbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpMbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0
3bidirectional/forward_lstm/while/lstm_cell_1/MatMulMatMulKbidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Jbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΥ
Dbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpObidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0π
5bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1MatMul.bidirectional_forward_lstm_while_placeholder_2Lbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θμ
0bidirectional/forward_lstm/while/lstm_cell_1/addAddV2=bidirectional/forward_lstm/while/lstm_cell_1/MatMul:product:0?bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΟ
Cbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpNbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0υ
4bidirectional/forward_lstm/while/lstm_cell_1/BiasAddBiasAdd4bidirectional/forward_lstm/while/lstm_cell_1/add:z:0Kbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ~
<bidirectional/forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :½
2bidirectional/forward_lstm/while/lstm_cell_1/splitSplitEbidirectional/forward_lstm/while/lstm_cell_1/split/split_dim:output:0=bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split?
4bidirectional/forward_lstm/while/lstm_cell_1/SigmoidSigmoid;bidirectional/forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2°
6bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_1Sigmoid;bidirectional/forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2Υ
0bidirectional/forward_lstm/while/lstm_cell_1/mulMul:bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_1:y:0.bidirectional_forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2¨
1bidirectional/forward_lstm/while/lstm_cell_1/ReluRelu;bidirectional/forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2ζ
2bidirectional/forward_lstm/while/lstm_cell_1/mul_1Mul8bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid:y:0?bidirectional/forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2Ϋ
2bidirectional/forward_lstm/while/lstm_cell_1/add_1AddV24bidirectional/forward_lstm/while/lstm_cell_1/mul:z:06bidirectional/forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2°
6bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_2Sigmoid;bidirectional/forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2₯
3bidirectional/forward_lstm/while/lstm_cell_1/Relu_1Relu6bidirectional/forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2κ
2bidirectional/forward_lstm/while/lstm_cell_1/mul_2Mul:bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_2:y:0Abidirectional/forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2°
Ebidirectional/forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.bidirectional_forward_lstm_while_placeholder_1,bidirectional_forward_lstm_while_placeholder6bidirectional/forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?h
&bidirectional/forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :­
$bidirectional/forward_lstm/while/addAddV2,bidirectional_forward_lstm_while_placeholder/bidirectional/forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: j
(bidirectional/forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Σ
&bidirectional/forward_lstm/while/add_1AddV2Nbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_loop_counter1bidirectional/forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: ͺ
)bidirectional/forward_lstm/while/IdentityIdentity*bidirectional/forward_lstm/while/add_1:z:0&^bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: Φ
+bidirectional/forward_lstm/while/Identity_1IdentityTbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_maximum_iterations&^bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: ͺ
+bidirectional/forward_lstm/while/Identity_2Identity(bidirectional/forward_lstm/while/add:z:0&^bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: κ
+bidirectional/forward_lstm/while/Identity_3IdentityUbidirectional/forward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?Ι
+bidirectional/forward_lstm/while/Identity_4Identity6bidirectional/forward_lstm/while/lstm_cell_1/mul_2:z:0&^bidirectional/forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2Ι
+bidirectional/forward_lstm/while/Identity_5Identity6bidirectional/forward_lstm/while/lstm_cell_1/add_1:z:0&^bidirectional/forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2Ή
%bidirectional/forward_lstm/while/NoOpNoOpD^bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpC^bidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpE^bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Kbidirectional_forward_lstm_while_bidirectional_forward_lstm_strided_slice_1Mbidirectional_forward_lstm_while_bidirectional_forward_lstm_strided_slice_1_0"_
)bidirectional_forward_lstm_while_identity2bidirectional/forward_lstm/while/Identity:output:0"c
+bidirectional_forward_lstm_while_identity_14bidirectional/forward_lstm/while/Identity_1:output:0"c
+bidirectional_forward_lstm_while_identity_24bidirectional/forward_lstm/while/Identity_2:output:0"c
+bidirectional_forward_lstm_while_identity_34bidirectional/forward_lstm/while/Identity_3:output:0"c
+bidirectional_forward_lstm_while_identity_44bidirectional/forward_lstm/while/Identity_4:output:0"c
+bidirectional_forward_lstm_while_identity_54bidirectional/forward_lstm/while/Identity_5:output:0"
Lbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resourceNbidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0" 
Mbidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceObidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"
Kbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resourceMbidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"
bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensorbidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2
Cbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpCbidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2
Bbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpBbidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2
Dbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpDbidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ϊ
Ά	
6sequential_bidirectional_forward_lstm_while_cond_37864h
dsequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_loop_countern
jsequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_maximum_iterations;
7sequential_bidirectional_forward_lstm_while_placeholder=
9sequential_bidirectional_forward_lstm_while_placeholder_1=
9sequential_bidirectional_forward_lstm_while_placeholder_2=
9sequential_bidirectional_forward_lstm_while_placeholder_3j
fsequential_bidirectional_forward_lstm_while_less_sequential_bidirectional_forward_lstm_strided_slice_1
{sequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_cond_37864___redundant_placeholder0
{sequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_cond_37864___redundant_placeholder1
{sequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_cond_37864___redundant_placeholder2
{sequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_cond_37864___redundant_placeholder38
4sequential_bidirectional_forward_lstm_while_identity
ϊ
0sequential/bidirectional/forward_lstm/while/LessLess7sequential_bidirectional_forward_lstm_while_placeholderfsequential_bidirectional_forward_lstm_while_less_sequential_bidirectional_forward_lstm_strided_slice_1*
T0*
_output_shapes
: 
4sequential/bidirectional/forward_lstm/while/IdentityIdentity4sequential/bidirectional/forward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "u
4sequential_bidirectional_forward_lstm_while_identity=sequential/bidirectional/forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ν


*__inference_sequential_layer_call_fn_39882
bidirectional_input
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
	unknown_5:dP
	unknown_6:P
	unknown_7:PP
	unknown_8:P
identity’StatefulPartitionedCallΟ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_39859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????P
-
_user_specified_namebidirectional_input
	

-__inference_bidirectional_layer_call_fn_41154

inputs
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_39798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
γ7
Ζ
while_body_42779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_1_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_1_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_1/BiasAdd/ReadVariableOp’'while/lstm_cell_1/MatMul/ReadVariableOp’)while/lstm_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????―
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0
'while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_1/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_1/addAddV2"while/lstm_cell_1/MatMul:product:0$while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_1/BiasAddBiasAddwhile/lstm_cell_1/add:z:00while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0"while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_1/SigmoidSigmoid while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_1Sigmoid while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mulMulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_1/ReluRelu while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_1Mulwhile/lstm_cell_1/Sigmoid:y:0$while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/add_1AddV2while/lstm_cell_1/mul:z:0while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_1/Sigmoid_2Sigmoid while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_1/Relu_1Reluwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_1/mul_2Mulwhile/lstm_cell_1/Sigmoid_2:y:0&while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_1/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_1/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_1/BiasAdd/ReadVariableOp(^while/lstm_cell_1/MatMul/ReadVariableOp*^while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_1_biasadd_readvariableop_resource3while_lstm_cell_1_biasadd_readvariableop_resource_0"j
2while_lstm_cell_1_matmul_1_readvariableop_resource4while_lstm_cell_1_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_1_matmul_readvariableop_resource2while_lstm_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_1/BiasAdd/ReadVariableOp(while/lstm_cell_1/BiasAdd/ReadVariableOp2R
'while/lstm_cell_1/MatMul/ReadVariableOp'while/lstm_cell_1/MatMul/ReadVariableOp2V
)while/lstm_cell_1/MatMul_1/ReadVariableOp)while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Ϋ

F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_43728

inputs
states_0
states_11
matmul_readvariableop_resource:	Θ3
 matmul_1_readvariableop_resource:	2Θ.
biasadd_readvariableop_resource:	Θ
identity

identity_1

identity_2’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ά
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
Β
ξ
,bidirectional_backward_lstm_while_cond_40668T
Pbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_loop_counterZ
Vbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_maximum_iterations1
-bidirectional_backward_lstm_while_placeholder3
/bidirectional_backward_lstm_while_placeholder_13
/bidirectional_backward_lstm_while_placeholder_23
/bidirectional_backward_lstm_while_placeholder_3V
Rbidirectional_backward_lstm_while_less_bidirectional_backward_lstm_strided_slice_1k
gbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_cond_40668___redundant_placeholder0k
gbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_cond_40668___redundant_placeholder1k
gbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_cond_40668___redundant_placeholder2k
gbidirectional_backward_lstm_while_bidirectional_backward_lstm_while_cond_40668___redundant_placeholder3.
*bidirectional_backward_lstm_while_identity
?
&bidirectional/backward_lstm/while/LessLess-bidirectional_backward_lstm_while_placeholderRbidirectional_backward_lstm_while_less_bidirectional_backward_lstm_strided_slice_1*
T0*
_output_shapes
: 
*bidirectional/backward_lstm/while/IdentityIdentity*bidirectional/backward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "a
*bidirectional_backward_lstm_while_identity3bidirectional/backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
γ7
Ζ
while_body_43546
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_2_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_2/BiasAdd/ReadVariableOp’'while/lstm_cell_2/MatMul/ReadVariableOp’)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????―
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
γ7
Ζ
while_body_39211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘG
4while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘB
3while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_2_matmul_readvariableop_resource:	ΘE
2while_lstm_cell_2_matmul_1_readvariableop_resource:	2Θ@
1while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’(while/lstm_cell_2/BiasAdd/ReadVariableOp’'while/lstm_cell_2/MatMul/ReadVariableOp’)while/lstm_cell_2/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????―
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0
'while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0Έ
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
)while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
while/lstm_cell_2/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
while/lstm_cell_2/addAddV2"while/lstm_cell_2/MatMul:product:0$while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
(while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0€
while/lstm_cell_2/BiasAddBiasAddwhile/lstm_cell_2/add:z:00while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θc
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :μ
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0"while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitx
while/lstm_cell_2/SigmoidSigmoid while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_1Sigmoid while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mulMulwhile/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2r
while/lstm_cell_2/ReluRelu while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_1Mulwhile/lstm_cell_2/Sigmoid:y:0$while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/add_1AddV2while/lstm_cell_2/mul:z:0while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2z
while/lstm_cell_2/Sigmoid_2Sigmoid while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2o
while/lstm_cell_2/Relu_1Reluwhile/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_2/mul_2Mulwhile/lstm_cell_2/Sigmoid_2:y:0&while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2Δ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?x
while/Identity_4Identitywhile/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2x
while/Identity_5Identitywhile/lstm_cell_2/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:?????????2Ν

while/NoOpNoOp)^while/lstm_cell_2/BiasAdd/ReadVariableOp(^while/lstm_cell_2/MatMul/ReadVariableOp*^while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_2_biasadd_readvariableop_resource3while_lstm_cell_2_biasadd_readvariableop_resource_0"j
2while_lstm_cell_2_matmul_1_readvariableop_resource4while_lstm_cell_2_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_2_matmul_readvariableop_resource2while_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2T
(while/lstm_cell_2/BiasAdd/ReadVariableOp(while/lstm_cell_2/BiasAdd/ReadVariableOp2R
'while/lstm_cell_2/MatMul/ReadVariableOp'while/lstm_cell_2/MatMul/ReadVariableOp2V
)while/lstm_cell_2/MatMul_1/ReadVariableOp)while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Π"
Υ
while_body_38378
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_1_38402_0:	Θ,
while_lstm_cell_1_38404_0:	2Θ(
while_lstm_cell_1_38406_0:	Θ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_1_38402:	Θ*
while_lstm_cell_1_38404:	2Θ&
while_lstm_cell_1_38406:	Θ’)while/lstm_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ͺ
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_38402_0while_lstm_cell_1_38404_0while_lstm_cell_1_38406_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_38319Ϋ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:ιθ?M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :ιθ?
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????2
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????2x

while/NoOpNoOp*^while/lstm_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_1_38402while_lstm_cell_1_38402_0"4
while_lstm_cell_1_38404while_lstm_cell_1_38404_0"4
while_lstm_cell_1_38406while_lstm_cell_1_38406_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
°
Ύ
while_cond_42492
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_42492___redundant_placeholder03
/while_while_cond_42492___redundant_placeholder13
/while_while_cond_42492___redundant_placeholder23
/while_while_cond_42492___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ή
Φ
backward_lstm_while_cond_401498
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_3:
6backward_lstm_while_less_backward_lstm_strided_slice_1O
Kbackward_lstm_while_backward_lstm_while_cond_40149___redundant_placeholder0O
Kbackward_lstm_while_backward_lstm_while_cond_40149___redundant_placeholder1O
Kbackward_lstm_while_backward_lstm_while_cond_40149___redundant_placeholder2O
Kbackward_lstm_while_backward_lstm_while_cond_40149___redundant_placeholder3 
backward_lstm_while_identity

backward_lstm/while/LessLessbackward_lstm_while_placeholder6backward_lstm_while_less_backward_lstm_strided_slice_1*
T0*
_output_shapes
: g
backward_lstm/while/IdentityIdentitybackward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Ϋ
»
E__inference_sequential_layer_call_and_return_conditional_losses_40384
bidirectional_input&
bidirectional_40358:	Θ&
bidirectional_40360:	2Θ"
bidirectional_40362:	Θ&
bidirectional_40364:	Θ&
bidirectional_40366:	2Θ"
bidirectional_40368:	Θ
dense_40371:dP
dense_40373:P
dense_1_40378:PP
dense_1_40380:P
identity’%bidirectional/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCallκ
%bidirectional/StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputbidirectional_40358bidirectional_40360bidirectional_40362bidirectional_40364bidirectional_40366bidirectional_40368*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_39798
dense/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_40371dense_40373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_39822Ϋ
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_39833?
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_39840
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_40378dense_1_40380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_39852w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P°
NoOpNoOp&^bidirectional/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:` \
+
_output_shapes
:?????????P
-
_user_specified_namebidirectional_input
΄
?
E__inference_sequential_layer_call_and_return_conditional_losses_39859

inputs&
bidirectional_39799:	Θ&
bidirectional_39801:	2Θ"
bidirectional_39803:	Θ&
bidirectional_39805:	Θ&
bidirectional_39807:	2Θ"
bidirectional_39809:	Θ
dense_39823:dP
dense_39825:P
dense_1_39853:PP
dense_1_39855:P
identity’%bidirectional/StatefulPartitionedCall’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCallέ
%bidirectional/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_39799bidirectional_39801bidirectional_39803bidirectional_39805bidirectional_39807bidirectional_39809*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_39798
dense/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_39823dense_39825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_39822Ϋ
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_39833?
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_39840
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_39853dense_1_39855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_39852w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P°
NoOpNoOp&^bidirectional/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
©L

H__inference_backward_lstm_layer_call_and_return_conditional_losses_39295

inputs=
*lstm_cell_2_matmul_readvariableop_resource:	Θ?
,lstm_cell_2_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_2/BiasAdd/ReadVariableOp’!lstm_cell_2/MatMul/ReadVariableOp’#lstm_cell_2/MatMul_1/ReadVariableOp’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????ε
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_2/MatMul_1MatMulzeros:output:0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_2/ReluRelulstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_2/Relu_1Relulstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0 lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_2_matmul_readvariableop_resource,lstm_cell_2_matmul_1_readvariableop_resource+lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39211*
condR
while_cond_39210*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
°
Ύ
while_cond_42921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_42921___redundant_placeholder03
/while_while_cond_42921___redundant_placeholder13
/while_while_cond_42921___redundant_placeholder23
/while_while_cond_42921___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
°
Ύ
while_cond_42635
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_42635___redundant_placeholder03
/while_while_cond_42635___redundant_placeholder13
/while_while_cond_42635___redundant_placeholder23
/while_while_cond_42635___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
°
Ύ
while_cond_38377
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38377___redundant_placeholder03
/while_while_cond_38377___redundant_placeholder13
/while_while_cond_38377___redundant_placeholder23
/while_while_cond_38377___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
Φ9

H__inference_backward_lstm_layer_call_and_return_conditional_losses_38801

inputs$
lstm_cell_2_38719:	Θ$
lstm_cell_2_38721:	2Θ 
lstm_cell_2_38723:	Θ
identity’#lstm_cell_2/StatefulPartitionedCall’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ε
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskμ
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_38719lstm_cell_2_38721lstm_cell_2_38723*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_38671n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ―
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_38719lstm_cell_2_38721lstm_cell_2_38723*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38732*
condR
while_cond_38731*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2t
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
°
Ύ
while_cond_43255
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_43255___redundant_placeholder03
/while_while_cond_43255___redundant_placeholder13
/while_while_cond_43255___redundant_placeholder23
/while_while_cond_43255___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
°
Ύ
while_cond_38538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38538___redundant_placeholder03
/while_while_cond_38538___redundant_placeholder13
/while_while_cond_38538___redundant_placeholder23
/while_while_cond_38538___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
°
Ύ
while_cond_38731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38731___redundant_placeholder03
/while_while_cond_38731___redundant_placeholder13
/while_while_cond_38731___redundant_placeholder23
/while_while_cond_38731___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
£
Ή
,__inference_forward_lstm_layer_call_fn_42434

inputs
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity’StatefulPartitionedCallι
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_forward_lstm_layer_call_and_return_conditional_losses_39460o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
¦H

backward_lstm_while_body_416578
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_37
3backward_lstm_while_backward_lstm_strided_slice_1_0s
obackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0S
@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘU
Bbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘP
Abackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ 
backward_lstm_while_identity"
backward_lstm_while_identity_1"
backward_lstm_while_identity_2"
backward_lstm_while_identity_3"
backward_lstm_while_identity_4"
backward_lstm_while_identity_55
1backward_lstm_while_backward_lstm_strided_slice_1q
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorQ
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	ΘS
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘN
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp
Ebackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????υ
7backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0backward_lstm_while_placeholderNbackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0·
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0β
&backward_lstm/while/lstm_cell_2/MatMulMatMul>backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ»
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ι
(backward_lstm/while/lstm_cell_2/MatMul_1MatMul!backward_lstm_while_placeholder_2?backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΕ
#backward_lstm/while/lstm_cell_2/addAddV20backward_lstm/while/lstm_cell_2/MatMul:product:02backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ΅
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Ξ
'backward_lstm/while/lstm_cell_2/BiasAddBiasAdd'backward_lstm/while/lstm_cell_2/add:z:0>backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θq
/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%backward_lstm/while/lstm_cell_2/splitSplit8backward_lstm/while/lstm_cell_2/split/split_dim:output:00backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
'backward_lstm/while/lstm_cell_2/SigmoidSigmoid.backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_1Sigmoid.backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2?
#backward_lstm/while/lstm_cell_2/mulMul-backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0!backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
$backward_lstm/while/lstm_cell_2/ReluRelu.backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2Ώ
%backward_lstm/while/lstm_cell_2/mul_1Mul+backward_lstm/while/lstm_cell_2/Sigmoid:y:02backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2΄
%backward_lstm/while/lstm_cell_2/add_1AddV2'backward_lstm/while/lstm_cell_2/mul:z:0)backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_2Sigmoid.backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
&backward_lstm/while/lstm_cell_2/Relu_1Relu)backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2Γ
%backward_lstm/while/lstm_cell_2/mul_2Mul-backward_lstm/while/lstm_cell_2/Sigmoid_2:y:04backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ό
8backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!backward_lstm_while_placeholder_1backward_lstm_while_placeholder)backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?[
backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/addAddV2backward_lstm_while_placeholder"backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: ]
backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/add_1AddV24backward_lstm_while_backward_lstm_while_loop_counter$backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_lstm/while/IdentityIdentitybackward_lstm/while/add_1:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: ’
backward_lstm/while/Identity_1Identity:backward_lstm_while_backward_lstm_while_maximum_iterations^backward_lstm/while/NoOp*
T0*
_output_shapes
: 
backward_lstm/while/Identity_2Identitybackward_lstm/while/add:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: Γ
backward_lstm/while/Identity_3IdentityHbackward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?’
backward_lstm/while/Identity_4Identity)backward_lstm/while/lstm_cell_2/mul_2:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2’
backward_lstm/while/Identity_5Identity)backward_lstm/while/lstm_cell_2/add_1:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
backward_lstm/while/NoOpNoOp7^backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6^backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp8^backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1backward_lstm_while_backward_lstm_strided_slice_13backward_lstm_while_backward_lstm_strided_slice_1_0"E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0"I
backward_lstm_while_identity_1'backward_lstm/while/Identity_1:output:0"I
backward_lstm_while_identity_2'backward_lstm/while/Identity_2:output:0"I
backward_lstm_while_identity_3'backward_lstm/while/Identity_3:output:0"I
backward_lstm_while_identity_4'backward_lstm/while/Identity_4:output:0"I
backward_lstm_while_identity_5'backward_lstm/while/Identity_5:output:0"
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resourceBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"ΰ
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2p
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2n
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2r
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
¦

ω
*__inference_sequential_layer_call_fn_40444

inputs
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
	unknown_5:dP
	unknown_6:P
	unknown_7:PP
	unknown_8:P
identity’StatefulPartitionedCallΒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_39859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
Ήd

6sequential_bidirectional_forward_lstm_while_body_37865h
dsequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_loop_countern
jsequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_maximum_iterations;
7sequential_bidirectional_forward_lstm_while_placeholder=
9sequential_bidirectional_forward_lstm_while_placeholder_1=
9sequential_bidirectional_forward_lstm_while_placeholder_2=
9sequential_bidirectional_forward_lstm_while_placeholder_3g
csequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_strided_slice_1_0€
sequential_bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0k
Xsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0:	Θm
Zsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0:	2Θh
Ysequential_bidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0:	Θ8
4sequential_bidirectional_forward_lstm_while_identity:
6sequential_bidirectional_forward_lstm_while_identity_1:
6sequential_bidirectional_forward_lstm_while_identity_2:
6sequential_bidirectional_forward_lstm_while_identity_3:
6sequential_bidirectional_forward_lstm_while_identity_4:
6sequential_bidirectional_forward_lstm_while_identity_5e
asequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_strided_slice_1’
sequential_bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensori
Vsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource:	Θk
Xsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource:	2Θf
Wsequential_bidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource:	Θ’Nsequential/bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp’Msequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp’Osequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp?
]sequential/bidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ε
Osequential/bidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_07sequential_bidirectional_forward_lstm_while_placeholderfsequential/bidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0η
Msequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpXsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0ͺ
>sequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMulMatMulVsequential/bidirectional/forward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Usequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θλ
Osequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpZsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0
@sequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1MatMul9sequential_bidirectional_forward_lstm_while_placeholder_2Wsequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
;sequential/bidirectional/forward_lstm/while/lstm_cell_1/addAddV2Hsequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul:product:0Jsequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θε
Nsequential/bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpYsequential_bidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0
?sequential/bidirectional/forward_lstm/while/lstm_cell_1/BiasAddBiasAdd?sequential/bidirectional/forward_lstm/while/lstm_cell_1/add:z:0Vsequential/bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
Gsequential/bidirectional/forward_lstm/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ή
=sequential/bidirectional/forward_lstm/while/lstm_cell_1/splitSplitPsequential/bidirectional/forward_lstm/while/lstm_cell_1/split/split_dim:output:0Hsequential/bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitΔ
?sequential/bidirectional/forward_lstm/while/lstm_cell_1/SigmoidSigmoidFsequential/bidirectional/forward_lstm/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2Ζ
Asequential/bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_1SigmoidFsequential/bidirectional/forward_lstm/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2φ
;sequential/bidirectional/forward_lstm/while/lstm_cell_1/mulMulEsequential/bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_1:y:09sequential_bidirectional_forward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2Ύ
<sequential/bidirectional/forward_lstm/while/lstm_cell_1/ReluReluFsequential/bidirectional/forward_lstm/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
=sequential/bidirectional/forward_lstm/while/lstm_cell_1/mul_1MulCsequential/bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid:y:0Jsequential/bidirectional/forward_lstm/while/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2ό
=sequential/bidirectional/forward_lstm/while/lstm_cell_1/add_1AddV2?sequential/bidirectional/forward_lstm/while/lstm_cell_1/mul:z:0Asequential/bidirectional/forward_lstm/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2Ζ
Asequential/bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_2SigmoidFsequential/bidirectional/forward_lstm/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2»
>sequential/bidirectional/forward_lstm/while/lstm_cell_1/Relu_1ReluAsequential/bidirectional/forward_lstm/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
=sequential/bidirectional/forward_lstm/while/lstm_cell_1/mul_2MulEsequential/bidirectional/forward_lstm/while/lstm_cell_1/Sigmoid_2:y:0Lsequential/bidirectional/forward_lstm/while/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ά
Psequential/bidirectional/forward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem9sequential_bidirectional_forward_lstm_while_placeholder_17sequential_bidirectional_forward_lstm_while_placeholderAsequential/bidirectional/forward_lstm/while/lstm_cell_1/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?s
1sequential/bidirectional/forward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ξ
/sequential/bidirectional/forward_lstm/while/addAddV27sequential_bidirectional_forward_lstm_while_placeholder:sequential/bidirectional/forward_lstm/while/add/y:output:0*
T0*
_output_shapes
: u
3sequential/bidirectional/forward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
1sequential/bidirectional/forward_lstm/while/add_1AddV2dsequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_loop_counter<sequential/bidirectional/forward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: Λ
4sequential/bidirectional/forward_lstm/while/IdentityIdentity5sequential/bidirectional/forward_lstm/while/add_1:z:01^sequential/bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: 
6sequential/bidirectional/forward_lstm/while/Identity_1Identityjsequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_while_maximum_iterations1^sequential/bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: Λ
6sequential/bidirectional/forward_lstm/while/Identity_2Identity3sequential/bidirectional/forward_lstm/while/add:z:01^sequential/bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: 
6sequential/bidirectional/forward_lstm/while/Identity_3Identity`sequential/bidirectional/forward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^sequential/bidirectional/forward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?κ
6sequential/bidirectional/forward_lstm/while/Identity_4IdentityAsequential/bidirectional/forward_lstm/while/lstm_cell_1/mul_2:z:01^sequential/bidirectional/forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2κ
6sequential/bidirectional/forward_lstm/while/Identity_5IdentityAsequential/bidirectional/forward_lstm/while/lstm_cell_1/add_1:z:01^sequential/bidirectional/forward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2ε
0sequential/bidirectional/forward_lstm/while/NoOpNoOpO^sequential/bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpN^sequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpP^sequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "u
4sequential_bidirectional_forward_lstm_while_identity=sequential/bidirectional/forward_lstm/while/Identity:output:0"y
6sequential_bidirectional_forward_lstm_while_identity_1?sequential/bidirectional/forward_lstm/while/Identity_1:output:0"y
6sequential_bidirectional_forward_lstm_while_identity_2?sequential/bidirectional/forward_lstm/while/Identity_2:output:0"y
6sequential_bidirectional_forward_lstm_while_identity_3?sequential/bidirectional/forward_lstm/while/Identity_3:output:0"y
6sequential_bidirectional_forward_lstm_while_identity_4?sequential/bidirectional/forward_lstm/while/Identity_4:output:0"y
6sequential_bidirectional_forward_lstm_while_identity_5?sequential/bidirectional/forward_lstm/while/Identity_5:output:0"΄
Wsequential_bidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resourceYsequential_bidirectional_forward_lstm_while_lstm_cell_1_biasadd_readvariableop_resource_0"Ά
Xsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resourceZsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_1_readvariableop_resource_0"²
Vsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resourceXsequential_bidirectional_forward_lstm_while_lstm_cell_1_matmul_readvariableop_resource_0"Θ
asequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_strided_slice_1csequential_bidirectional_forward_lstm_while_sequential_bidirectional_forward_lstm_strided_slice_1_0"Β
sequential_bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensorsequential_bidirectional_forward_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2 
Nsequential/bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOpNsequential/bidirectional/forward_lstm/while/lstm_cell_1/BiasAdd/ReadVariableOp2
Msequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOpMsequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul/ReadVariableOp2’
Osequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOpOsequential/bidirectional/forward_lstm/while/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
ϊ
Ξ	
7sequential_bidirectional_backward_lstm_while_cond_38005j
fsequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_loop_counterp
lsequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_maximum_iterations<
8sequential_bidirectional_backward_lstm_while_placeholder>
:sequential_bidirectional_backward_lstm_while_placeholder_1>
:sequential_bidirectional_backward_lstm_while_placeholder_2>
:sequential_bidirectional_backward_lstm_while_placeholder_3l
hsequential_bidirectional_backward_lstm_while_less_sequential_bidirectional_backward_lstm_strided_slice_1
}sequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_cond_38005___redundant_placeholder0
}sequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_cond_38005___redundant_placeholder1
}sequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_cond_38005___redundant_placeholder2
}sequential_bidirectional_backward_lstm_while_sequential_bidirectional_backward_lstm_while_cond_38005___redundant_placeholder39
5sequential_bidirectional_backward_lstm_while_identity
ώ
1sequential/bidirectional/backward_lstm/while/LessLess8sequential_bidirectional_backward_lstm_while_placeholderhsequential_bidirectional_backward_lstm_while_less_sequential_bidirectional_backward_lstm_strided_slice_1*
T0*
_output_shapes
: 
5sequential/bidirectional/backward_lstm/while/IdentityIdentity5sequential/bidirectional/backward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "w
5sequential_bidirectional_backward_lstm_while_identity>sequential/bidirectional/backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
μ

!__inference__traced_restore_44081
file_prefix/
assignvariableop_dense_kernel:dP+
assignvariableop_1_dense_bias:P3
!assignvariableop_2_dense_1_kernel:PP-
assignvariableop_3_dense_1_bias:P&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: S
@assignvariableop_9_bidirectional_forward_lstm_lstm_cell_1_kernel:	Θ^
Kassignvariableop_10_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel:	2ΘN
?assignvariableop_11_bidirectional_forward_lstm_lstm_cell_1_bias:	ΘU
Bassignvariableop_12_bidirectional_backward_lstm_lstm_cell_2_kernel:	Θ_
Lassignvariableop_13_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel:	2ΘO
@assignvariableop_14_bidirectional_backward_lstm_lstm_cell_2_bias:	Θ#
assignvariableop_15_total: #
assignvariableop_16_count: 9
'assignvariableop_17_adam_dense_kernel_m:dP3
%assignvariableop_18_adam_dense_bias_m:P;
)assignvariableop_19_adam_dense_1_kernel_m:PP5
'assignvariableop_20_adam_dense_1_bias_m:P[
Hassignvariableop_21_adam_bidirectional_forward_lstm_lstm_cell_1_kernel_m:	Θe
Rassignvariableop_22_adam_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_m:	2ΘU
Fassignvariableop_23_adam_bidirectional_forward_lstm_lstm_cell_1_bias_m:	Θ\
Iassignvariableop_24_adam_bidirectional_backward_lstm_lstm_cell_2_kernel_m:	Θf
Sassignvariableop_25_adam_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_m:	2ΘV
Gassignvariableop_26_adam_bidirectional_backward_lstm_lstm_cell_2_bias_m:	Θ9
'assignvariableop_27_adam_dense_kernel_v:dP3
%assignvariableop_28_adam_dense_bias_v:P;
)assignvariableop_29_adam_dense_1_kernel_v:PP5
'assignvariableop_30_adam_dense_1_bias_v:P[
Hassignvariableop_31_adam_bidirectional_forward_lstm_lstm_cell_1_kernel_v:	Θe
Rassignvariableop_32_adam_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_v:	2ΘU
Fassignvariableop_33_adam_bidirectional_forward_lstm_lstm_cell_1_bias_v:	Θ\
Iassignvariableop_34_adam_bidirectional_backward_lstm_lstm_cell_2_kernel_v:	Θf
Sassignvariableop_35_adam_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_v:	2ΘV
Gassignvariableop_36_adam_bidirectional_backward_lstm_lstm_cell_2_bias_v:	Θ
identity_38’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9ς
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*
valueB&B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHΌ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ί
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:―
AssignVariableOp_9AssignVariableOp@assignvariableop_9_bidirectional_forward_lstm_lstm_cell_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ό
AssignVariableOp_10AssignVariableOpKassignvariableop_10_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_11AssignVariableOp?assignvariableop_11_bidirectional_forward_lstm_lstm_cell_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_12AssignVariableOpBassignvariableop_12_bidirectional_backward_lstm_lstm_cell_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_13AssignVariableOpLassignvariableop_13_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_14AssignVariableOp@assignvariableop_14_bidirectional_backward_lstm_lstm_cell_2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ή
AssignVariableOp_21AssignVariableOpHassignvariableop_21_adam_bidirectional_forward_lstm_lstm_cell_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Γ
AssignVariableOp_22AssignVariableOpRassignvariableop_22_adam_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_23AssignVariableOpFassignvariableop_23_adam_bidirectional_forward_lstm_lstm_cell_1_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ί
AssignVariableOp_24AssignVariableOpIassignvariableop_24_adam_bidirectional_backward_lstm_lstm_cell_2_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Δ
AssignVariableOp_25AssignVariableOpSassignvariableop_25_adam_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Έ
AssignVariableOp_26AssignVariableOpGassignvariableop_26_adam_bidirectional_backward_lstm_lstm_cell_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ή
AssignVariableOp_31AssignVariableOpHassignvariableop_31_adam_bidirectional_forward_lstm_lstm_cell_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Γ
AssignVariableOp_32AssignVariableOpRassignvariableop_32_adam_bidirectional_forward_lstm_lstm_cell_1_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_33AssignVariableOpFassignvariableop_33_adam_bidirectional_forward_lstm_lstm_cell_1_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ί
AssignVariableOp_34AssignVariableOpIassignvariableop_34_adam_bidirectional_backward_lstm_lstm_cell_2_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Δ
AssignVariableOp_35AssignVariableOpSassignvariableop_35_adam_bidirectional_backward_lstm_lstm_cell_2_recurrent_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Έ
AssignVariableOp_36AssignVariableOpGassignvariableop_36_adam_bidirectional_backward_lstm_lstm_cell_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ύ
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: κ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
H

backward_lstm_while_body_419438
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_37
3backward_lstm_while_backward_lstm_strided_slice_1_0s
obackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0S
@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘU
Bbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘP
Abackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ 
backward_lstm_while_identity"
backward_lstm_while_identity_1"
backward_lstm_while_identity_2"
backward_lstm_while_identity_3"
backward_lstm_while_identity_4"
backward_lstm_while_identity_55
1backward_lstm_while_backward_lstm_strided_slice_1q
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorQ
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	ΘS
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘN
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp
Ebackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   μ
7backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0backward_lstm_while_placeholderNbackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0·
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0β
&backward_lstm/while/lstm_cell_2/MatMulMatMul>backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ»
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ι
(backward_lstm/while/lstm_cell_2/MatMul_1MatMul!backward_lstm_while_placeholder_2?backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΕ
#backward_lstm/while/lstm_cell_2/addAddV20backward_lstm/while/lstm_cell_2/MatMul:product:02backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ΅
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Ξ
'backward_lstm/while/lstm_cell_2/BiasAddBiasAdd'backward_lstm/while/lstm_cell_2/add:z:0>backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θq
/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%backward_lstm/while/lstm_cell_2/splitSplit8backward_lstm/while/lstm_cell_2/split/split_dim:output:00backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
'backward_lstm/while/lstm_cell_2/SigmoidSigmoid.backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_1Sigmoid.backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2?
#backward_lstm/while/lstm_cell_2/mulMul-backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0!backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
$backward_lstm/while/lstm_cell_2/ReluRelu.backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2Ώ
%backward_lstm/while/lstm_cell_2/mul_1Mul+backward_lstm/while/lstm_cell_2/Sigmoid:y:02backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2΄
%backward_lstm/while/lstm_cell_2/add_1AddV2'backward_lstm/while/lstm_cell_2/mul:z:0)backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_2Sigmoid.backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
&backward_lstm/while/lstm_cell_2/Relu_1Relu)backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2Γ
%backward_lstm/while/lstm_cell_2/mul_2Mul-backward_lstm/while/lstm_cell_2/Sigmoid_2:y:04backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ό
8backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!backward_lstm_while_placeholder_1backward_lstm_while_placeholder)backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?[
backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/addAddV2backward_lstm_while_placeholder"backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: ]
backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/add_1AddV24backward_lstm_while_backward_lstm_while_loop_counter$backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_lstm/while/IdentityIdentitybackward_lstm/while/add_1:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: ’
backward_lstm/while/Identity_1Identity:backward_lstm_while_backward_lstm_while_maximum_iterations^backward_lstm/while/NoOp*
T0*
_output_shapes
: 
backward_lstm/while/Identity_2Identitybackward_lstm/while/add:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: Γ
backward_lstm/while/Identity_3IdentityHbackward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?’
backward_lstm/while/Identity_4Identity)backward_lstm/while/lstm_cell_2/mul_2:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2’
backward_lstm/while/Identity_5Identity)backward_lstm/while/lstm_cell_2/add_1:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
backward_lstm/while/NoOpNoOp7^backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6^backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp8^backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1backward_lstm_while_backward_lstm_strided_slice_13backward_lstm_while_backward_lstm_strided_slice_1_0"E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0"I
backward_lstm_while_identity_1'backward_lstm/while/Identity_1:output:0"I
backward_lstm_while_identity_2'backward_lstm/while/Identity_2:output:0"I
backward_lstm_while_identity_3'backward_lstm/while/Identity_3:output:0"I
backward_lstm_while_identity_4'backward_lstm/while/Identity_4:output:0"I
backward_lstm_while_identity_5'backward_lstm/while/Identity_5:output:0"
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resourceBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"ΰ
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2p
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2n
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2r
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Υ
`
B__inference_dropout_layer_call_and_return_conditional_losses_39840

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????P[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs

C
'__inference_dropout_layer_call_fn_42349

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_39840`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
Γ	
ρ
@__inference_dense_layer_call_and_return_conditional_losses_42334

inputs0
matmul_readvariableop_resource:dP-
biasadd_readvariableop_resource:P
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Σ

F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_38173

inputs

states
states_11
matmul_readvariableop_resource:	Θ3
 matmul_1_readvariableop_resource:	2Θ.
biasadd_readvariableop_resource:	Θ
identity

identity_1

identity_2’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ά
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates

Ό
-__inference_backward_lstm_layer_call_fn_43017
inputs_0
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity’StatefulPartitionedCallμ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_38608o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
η
τ
+__inference_lstm_cell_2_layer_call_fn_43745

inputs
states_0
states_1
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
identity

identity_1

identity_2’StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_38525o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
H

backward_lstm_while_body_401508
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_37
3backward_lstm_while_backward_lstm_strided_slice_1_0s
obackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0S
@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘU
Bbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘP
Abackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ 
backward_lstm_while_identity"
backward_lstm_while_identity_1"
backward_lstm_while_identity_2"
backward_lstm_while_identity_3"
backward_lstm_while_identity_4"
backward_lstm_while_identity_55
1backward_lstm_while_backward_lstm_strided_slice_1q
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorQ
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	ΘS
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘN
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp
Ebackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   μ
7backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0backward_lstm_while_placeholderNbackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0·
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0β
&backward_lstm/while/lstm_cell_2/MatMulMatMul>backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ»
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ι
(backward_lstm/while/lstm_cell_2/MatMul_1MatMul!backward_lstm_while_placeholder_2?backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΕ
#backward_lstm/while/lstm_cell_2/addAddV20backward_lstm/while/lstm_cell_2/MatMul:product:02backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ΅
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Ξ
'backward_lstm/while/lstm_cell_2/BiasAddBiasAdd'backward_lstm/while/lstm_cell_2/add:z:0>backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θq
/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%backward_lstm/while/lstm_cell_2/splitSplit8backward_lstm/while/lstm_cell_2/split/split_dim:output:00backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
'backward_lstm/while/lstm_cell_2/SigmoidSigmoid.backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_1Sigmoid.backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2?
#backward_lstm/while/lstm_cell_2/mulMul-backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0!backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
$backward_lstm/while/lstm_cell_2/ReluRelu.backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2Ώ
%backward_lstm/while/lstm_cell_2/mul_1Mul+backward_lstm/while/lstm_cell_2/Sigmoid:y:02backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2΄
%backward_lstm/while/lstm_cell_2/add_1AddV2'backward_lstm/while/lstm_cell_2/mul:z:0)backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_2Sigmoid.backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
&backward_lstm/while/lstm_cell_2/Relu_1Relu)backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2Γ
%backward_lstm/while/lstm_cell_2/mul_2Mul-backward_lstm/while/lstm_cell_2/Sigmoid_2:y:04backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ό
8backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!backward_lstm_while_placeholder_1backward_lstm_while_placeholder)backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?[
backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/addAddV2backward_lstm_while_placeholder"backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: ]
backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/add_1AddV24backward_lstm_while_backward_lstm_while_loop_counter$backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_lstm/while/IdentityIdentitybackward_lstm/while/add_1:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: ’
backward_lstm/while/Identity_1Identity:backward_lstm_while_backward_lstm_while_maximum_iterations^backward_lstm/while/NoOp*
T0*
_output_shapes
: 
backward_lstm/while/Identity_2Identitybackward_lstm/while/add:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: Γ
backward_lstm/while/Identity_3IdentityHbackward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?’
backward_lstm/while/Identity_4Identity)backward_lstm/while/lstm_cell_2/mul_2:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2’
backward_lstm/while/Identity_5Identity)backward_lstm/while/lstm_cell_2/add_1:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
backward_lstm/while/NoOpNoOp7^backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6^backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp8^backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1backward_lstm_while_backward_lstm_strided_slice_13backward_lstm_while_backward_lstm_strided_slice_1_0"E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0"I
backward_lstm_while_identity_1'backward_lstm/while/Identity_1:output:0"I
backward_lstm_while_identity_2'backward_lstm/while/Identity_2:output:0"I
backward_lstm_while_identity_3'backward_lstm/while/Identity_3:output:0"I
backward_lstm_while_identity_4'backward_lstm/while/Identity_4:output:0"I
backward_lstm_while_identity_5'backward_lstm/while/Identity_5:output:0"
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resourceBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"ΰ
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2p
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2n
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2r
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
Σ

F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_38671

inputs

states
states_11
matmul_readvariableop_resource:	Θ3
 matmul_1_readvariableop_resource:	2Θ.
biasadd_readvariableop_resource:	Θ
identity

identity_1

identity_2’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ά
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
ς΄

H__inference_bidirectional_layer_call_and_return_conditional_losses_42029

inputsJ
7forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	ΘL
9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘG
8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	ΘK
8backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	ΘM
:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘH
9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	Θ
identity’0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’backward_lstm/while’/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’forward_lstm/whileH
forward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:j
 forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_sliceStridedSliceforward_lstm/Shape:output:0)forward_lstm/strided_slice/stack:output:0+forward_lstm/strided_slice/stack_1:output:0+forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros/packedPack#forward_lstm/strided_slice:output:0$forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zerosFill"forward_lstm/zeros/packed:output:0!forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2_
forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
forward_lstm/zeros_1/packedPack#forward_lstm/strided_slice:output:0&forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:_
forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_lstm/zeros_1Fill$forward_lstm/zeros_1/packed:output:0#forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2p
forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_lstm/transpose	Transposeinputs$forward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????^
forward_lstm/Shape_1Shapeforward_lstm/transpose:y:0*
T0*
_output_shapes
:l
"forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_lstm/strided_slice_1StridedSliceforward_lstm/Shape_1:output:0+forward_lstm/strided_slice_1/stack:output:0-forward_lstm/strided_slice_1/stack_1:output:0-forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????Ϋ
forward_lstm/TensorArrayV2TensorListReserve1forward_lstm/TensorArrayV2/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
Bforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
4forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm/transpose:y:0Kforward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?l
"forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ͺ
forward_lstm/strided_slice_2StridedSliceforward_lstm/transpose:y:0+forward_lstm/strided_slice_2/stack:output:0-forward_lstm/strided_slice_2/stack_1:output:0-forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask§
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp7forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0»
forward_lstm/lstm_cell_1/MatMulMatMul%forward_lstm/strided_slice_2:output:06forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ«
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0΅
!forward_lstm/lstm_cell_1/MatMul_1MatMulforward_lstm/zeros:output:08forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ°
forward_lstm/lstm_cell_1/addAddV2)forward_lstm/lstm_cell_1/MatMul:product:0+forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ₯
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ή
 forward_lstm/lstm_cell_1/BiasAddBiasAdd forward_lstm/lstm_cell_1/add:z:07forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θj
(forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
forward_lstm/lstm_cell_1/splitSplit1forward_lstm/lstm_cell_1/split/split_dim:output:0)forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
 forward_lstm/lstm_cell_1/SigmoidSigmoid'forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid'forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/mulMul&forward_lstm/lstm_cell_1/Sigmoid_1:y:0forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/ReluRelu'forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2ͺ
forward_lstm/lstm_cell_1/mul_1Mul$forward_lstm/lstm_cell_1/Sigmoid:y:0+forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
forward_lstm/lstm_cell_1/add_1AddV2 forward_lstm/lstm_cell_1/mul:z:0"forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
"forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid'forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2}
forward_lstm/lstm_cell_1/Relu_1Relu"forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2?
forward_lstm/lstm_cell_1/mul_2Mul&forward_lstm/lstm_cell_1/Sigmoid_2:y:0-forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2{
*forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ί
forward_lstm/TensorArrayV2_1TensorListReserve3forward_lstm/TensorArrayV2_1/element_shape:output:0%forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?S
forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
forward_lstm/whileWhile(forward_lstm/while/loop_counter:output:0.forward_lstm/while/maximum_iterations:output:0forward_lstm/time:output:0%forward_lstm/TensorArrayV2_1:handle:0forward_lstm/zeros:output:0forward_lstm/zeros_1:output:0%forward_lstm/strided_slice_1:output:0Dforward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07forward_lstm_lstm_cell_1_matmul_readvariableop_resource9forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource8forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
forward_lstm_while_body_41802*)
cond!R
forward_lstm_while_cond_41801*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
=forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ι
/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm/while:output:3Fforward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0u
"forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Θ
forward_lstm/strided_slice_3StridedSlice8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0+forward_lstm/strided_slice_3/stack:output:0-forward_lstm/strided_slice_3/stack_1:output:0-forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maskr
forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
forward_lstm/transpose_1	Transpose8forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0&forward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2h
forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    I
backward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:k
!backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_lstm/strided_sliceStridedSlicebackward_lstm/Shape:output:0*backward_lstm/strided_slice/stack:output:0,backward_lstm/strided_slice/stack_1:output:0,backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
backward_lstm/zeros/packedPack$backward_lstm/strided_slice:output:0%backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zerosFill#backward_lstm/zeros/packed:output:0"backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2`
backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2‘
backward_lstm/zeros_1/packedPack$backward_lstm/strided_slice:output:0'backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:`
backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_lstm/zeros_1Fill%backward_lstm/zeros_1/packed:output:0$backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2q
backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_lstm/transpose	Transposeinputs%backward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????`
backward_lstm/Shape_1Shapebackward_lstm/transpose:y:0*
T0*
_output_shapes
:m
#backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:‘
backward_lstm/strided_slice_1StridedSlicebackward_lstm/Shape_1:output:0,backward_lstm/strided_slice_1/stack:output:0.backward_lstm/strided_slice_1/stack_1:output:0.backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ή
backward_lstm/TensorArrayV2TensorListReserve2backward_lstm/TensorArrayV2/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?f
backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
backward_lstm/ReverseV2	ReverseV2backward_lstm/transpose:y:0%backward_lstm/ReverseV2/axis:output:0*
T0*+
_output_shapes
:P?????????
Cbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
5backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor backward_lstm/ReverseV2:output:0Lbackward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?m
#backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:―
backward_lstm/strided_slice_2StridedSlicebackward_lstm/transpose:y:0,backward_lstm/strided_slice_2/stack:output:0.backward_lstm/strided_slice_2/stack_1:output:0.backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask©
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp8backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0Ύ
 backward_lstm/lstm_cell_2/MatMulMatMul&backward_lstm/strided_slice_2:output:07backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ­
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0Έ
"backward_lstm/lstm_cell_2/MatMul_1MatMulbackward_lstm/zeros:output:09backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ³
backward_lstm/lstm_cell_2/addAddV2*backward_lstm/lstm_cell_2/MatMul:product:0,backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ§
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0Ό
!backward_lstm/lstm_cell_2/BiasAddBiasAdd!backward_lstm/lstm_cell_2/add:z:08backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θk
)backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/lstm_cell_2/splitSplit2backward_lstm/lstm_cell_2/split/split_dim:output:0*backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
!backward_lstm/lstm_cell_2/SigmoidSigmoid(backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_1Sigmoid(backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/mulMul'backward_lstm/lstm_cell_2/Sigmoid_1:y:0backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
backward_lstm/lstm_cell_2/ReluRelu(backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2­
backward_lstm/lstm_cell_2/mul_1Mul%backward_lstm/lstm_cell_2/Sigmoid:y:0,backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2’
backward_lstm/lstm_cell_2/add_1AddV2!backward_lstm/lstm_cell_2/mul:z:0#backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
#backward_lstm/lstm_cell_2/Sigmoid_2Sigmoid(backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
 backward_lstm/lstm_cell_2/Relu_1Relu#backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2±
backward_lstm/lstm_cell_2/mul_2Mul'backward_lstm/lstm_cell_2/Sigmoid_2:y:0.backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2|
+backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   β
backward_lstm/TensorArrayV2_1TensorListReserve4backward_lstm/TensorArrayV2_1/element_shape:output:0&backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?T
backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Α
backward_lstm/whileWhile)backward_lstm/while/loop_counter:output:0/backward_lstm/while/maximum_iterations:output:0backward_lstm/time:output:0&backward_lstm/TensorArrayV2_1:handle:0backward_lstm/zeros:output:0backward_lstm/zeros_1:output:0&backward_lstm/strided_slice_1:output:0Ebackward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08backward_lstm_lstm_cell_2_matmul_readvariableop_resource:backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource9backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
backward_lstm_while_body_41943**
cond"R 
backward_lstm_while_cond_41942*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
>backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   μ
0backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm/while:output:3Gbackward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0v
#backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ν
backward_lstm/strided_slice_3StridedSlice9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0,backward_lstm/strided_slice_3/stack:output:0.backward_lstm/strided_slice_3/stack_1:output:0.backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_masks
backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ΐ
backward_lstm/transpose_1	Transpose9backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0'backward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2i
backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concatConcatV2%forward_lstm/strided_slice_3:output:0&backward_lstm/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????d 
NoOpNoOp1^backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0^backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2^backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp^backward_lstm/while0^forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/^forward_lstm/lstm_cell_1/MatMul/ReadVariableOp1^forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp^forward_lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????P: : : : : : 2d
0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp0backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2b
/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2f
1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp1backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2*
backward_lstm/whilebackward_lstm/while2b
/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2`
.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp.forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2d
0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp0forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2(
forward_lstm/whileforward_lstm/while:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
¦H

backward_lstm_while_body_413718
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_37
3backward_lstm_while_backward_lstm_strided_slice_1_0s
obackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0S
@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0:	ΘU
Bbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0:	2ΘP
Abackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0:	Θ 
backward_lstm_while_identity"
backward_lstm_while_identity_1"
backward_lstm_while_identity_2"
backward_lstm_while_identity_3"
backward_lstm_while_identity_4"
backward_lstm_while_identity_55
1backward_lstm_while_backward_lstm_strided_slice_1q
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorQ
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource:	ΘS
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘN
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource:	Θ’6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp’5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp’7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp
Ebackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????υ
7backward_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0backward_lstm_while_placeholderNbackward_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype0·
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	Θ*
dtype0β
&backward_lstm/while/lstm_cell_2/MatMulMatMul>backward_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ»
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Θ*
dtype0Ι
(backward_lstm/while/lstm_cell_2/MatMul_1MatMul!backward_lstm_while_placeholder_2?backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΕ
#backward_lstm/while/lstm_cell_2/addAddV20backward_lstm/while/lstm_cell_2/MatMul:product:02backward_lstm/while/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ΅
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:Θ*
dtype0Ξ
'backward_lstm/while/lstm_cell_2/BiasAddBiasAdd'backward_lstm/while/lstm_cell_2/add:z:0>backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θq
/backward_lstm/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%backward_lstm/while/lstm_cell_2/splitSplit8backward_lstm/while/lstm_cell_2/split/split_dim:output:00backward_lstm/while/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split
'backward_lstm/while/lstm_cell_2/SigmoidSigmoid.backward_lstm/while/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_1Sigmoid.backward_lstm/while/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2?
#backward_lstm/while/lstm_cell_2/mulMul-backward_lstm/while/lstm_cell_2/Sigmoid_1:y:0!backward_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????2
$backward_lstm/while/lstm_cell_2/ReluRelu.backward_lstm/while/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2Ώ
%backward_lstm/while/lstm_cell_2/mul_1Mul+backward_lstm/while/lstm_cell_2/Sigmoid:y:02backward_lstm/while/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2΄
%backward_lstm/while/lstm_cell_2/add_1AddV2'backward_lstm/while/lstm_cell_2/mul:z:0)backward_lstm/while/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
)backward_lstm/while/lstm_cell_2/Sigmoid_2Sigmoid.backward_lstm/while/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
&backward_lstm/while/lstm_cell_2/Relu_1Relu)backward_lstm/while/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2Γ
%backward_lstm/while/lstm_cell_2/mul_2Mul-backward_lstm/while/lstm_cell_2/Sigmoid_2:y:04backward_lstm/while/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2ό
8backward_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!backward_lstm_while_placeholder_1backward_lstm_while_placeholder)backward_lstm/while/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:ιθ?[
backward_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/addAddV2backward_lstm_while_placeholder"backward_lstm/while/add/y:output:0*
T0*
_output_shapes
: ]
backward_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_lstm/while/add_1AddV24backward_lstm_while_backward_lstm_while_loop_counter$backward_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_lstm/while/IdentityIdentitybackward_lstm/while/add_1:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: ’
backward_lstm/while/Identity_1Identity:backward_lstm_while_backward_lstm_while_maximum_iterations^backward_lstm/while/NoOp*
T0*
_output_shapes
: 
backward_lstm/while/Identity_2Identitybackward_lstm/while/add:z:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: Γ
backward_lstm/while/Identity_3IdentityHbackward_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm/while/NoOp*
T0*
_output_shapes
: :ιθ?’
backward_lstm/while/Identity_4Identity)backward_lstm/while/lstm_cell_2/mul_2:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2’
backward_lstm/while/Identity_5Identity)backward_lstm/while/lstm_cell_2/add_1:z:0^backward_lstm/while/NoOp*
T0*'
_output_shapes
:?????????2
backward_lstm/while/NoOpNoOp7^backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6^backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp8^backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1backward_lstm_while_backward_lstm_strided_slice_13backward_lstm_while_backward_lstm_strided_slice_1_0"E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0"I
backward_lstm_while_identity_1'backward_lstm/while/Identity_1:output:0"I
backward_lstm_while_identity_2'backward_lstm/while/Identity_2:output:0"I
backward_lstm_while_identity_3'backward_lstm/while/Identity_3:output:0"I
backward_lstm_while_identity_4'backward_lstm/while/Identity_4:output:0"I
backward_lstm_while_identity_5'backward_lstm/while/Identity_5:output:0"
?backward_lstm_while_lstm_cell_2_biasadd_readvariableop_resourceAbackward_lstm_while_lstm_cell_2_biasadd_readvariableop_resource_0"
@backward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resourceBbackward_lstm_while_lstm_cell_2_matmul_1_readvariableop_resource_0"
>backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource@backward_lstm_while_lstm_cell_2_matmul_readvariableop_resource_0"ΰ
mbackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensorobackward_lstm_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????2:?????????2: : : : : 2p
6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp6backward_lstm/while/lstm_cell_2/BiasAdd/ReadVariableOp2n
5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp5backward_lstm/while/lstm_cell_2/MatMul/ReadVariableOp2r
7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp7backward_lstm/while/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
°
Ύ
while_cond_38186
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38186___redundant_placeholder03
/while_while_cond_38186___redundant_placeholder13
/while_while_cond_38186___redundant_placeholder23
/while_while_cond_38186___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:

Β
forward_lstm_while_cond_420876
2forward_lstm_while_forward_lstm_while_loop_counter<
8forward_lstm_while_forward_lstm_while_maximum_iterations"
forward_lstm_while_placeholder$
 forward_lstm_while_placeholder_1$
 forward_lstm_while_placeholder_2$
 forward_lstm_while_placeholder_38
4forward_lstm_while_less_forward_lstm_strided_slice_1M
Iforward_lstm_while_forward_lstm_while_cond_42087___redundant_placeholder0M
Iforward_lstm_while_forward_lstm_while_cond_42087___redundant_placeholder1M
Iforward_lstm_while_forward_lstm_while_cond_42087___redundant_placeholder2M
Iforward_lstm_while_forward_lstm_while_cond_42087___redundant_placeholder3
forward_lstm_while_identity

forward_lstm/while/LessLessforward_lstm_while_placeholder4forward_lstm_while_less_forward_lstm_strided_slice_1*
T0*
_output_shapes
: e
forward_lstm/while/IdentityIdentityforward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "C
forward_lstm_while_identity$forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
¦
Ϊ
+bidirectional_forward_lstm_while_cond_40827R
Nbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_loop_counterX
Tbidirectional_forward_lstm_while_bidirectional_forward_lstm_while_maximum_iterations0
,bidirectional_forward_lstm_while_placeholder2
.bidirectional_forward_lstm_while_placeholder_12
.bidirectional_forward_lstm_while_placeholder_22
.bidirectional_forward_lstm_while_placeholder_3T
Pbidirectional_forward_lstm_while_less_bidirectional_forward_lstm_strided_slice_1i
ebidirectional_forward_lstm_while_bidirectional_forward_lstm_while_cond_40827___redundant_placeholder0i
ebidirectional_forward_lstm_while_bidirectional_forward_lstm_while_cond_40827___redundant_placeholder1i
ebidirectional_forward_lstm_while_bidirectional_forward_lstm_while_cond_40827___redundant_placeholder2i
ebidirectional_forward_lstm_while_bidirectional_forward_lstm_while_cond_40827___redundant_placeholder3-
)bidirectional_forward_lstm_while_identity
Ξ
%bidirectional/forward_lstm/while/LessLess,bidirectional_forward_lstm_while_placeholderPbidirectional_forward_lstm_while_less_bidirectional_forward_lstm_strided_slice_1*
T0*
_output_shapes
: 
)bidirectional/forward_lstm/while/IdentityIdentity)bidirectional/forward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "_
)bidirectional_forward_lstm_while_identity2bidirectional/forward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
ΐJ

G__inference_forward_lstm_layer_call_and_return_conditional_losses_43006

inputs=
*lstm_cell_1_matmul_readvariableop_resource:	Θ?
,lstm_cell_1_matmul_1_readvariableop_resource:	2Θ:
+lstm_cell_1_biasadd_readvariableop_resource:	Θ
identity’"lstm_cell_1/BiasAdd/ReadVariableOp’!lstm_cell_1/MatMul/ReadVariableOp’#lstm_cell_1/MatMul_1/ReadVariableOp’while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????΄
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????ΰ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0
lstm_cell_1/MatMul_1MatMulzeros:output:0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θ
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ϊ
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitl
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2u
lstm_cell_1/mulMullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2f
lstm_cell_1/ReluRelulstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2x
lstm_cell_1/add_1AddV2lstm_cell_1/mul:z:0lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2n
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2c
lstm_cell_1/Relu_1Relulstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid_2:y:0 lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Έ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ύ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_1_matmul_readvariableop_resource,lstm_cell_1_matmul_1_readvariableop_resource+lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_42922*
condR
while_cond_42921*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   Λ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????2½
NoOpNoOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	

-__inference_bidirectional_layer_call_fn_41120
inputs_0
unknown:	Θ
	unknown_0:	2Θ
	unknown_1:	Θ
	unknown_2:	Θ
	unknown_3:	2Θ
	unknown_4:	Θ
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_39122o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
ͺο
§
E__inference_sequential_layer_call_and_return_conditional_losses_41076

inputsX
Ebidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resource:	ΘZ
Gbidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource:	2ΘU
Fbidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource:	ΘY
Fbidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resource:	Θ[
Hbidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource:	2ΘV
Gbidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource:	Θ6
$dense_matmul_readvariableop_resource:dP3
%dense_biasadd_readvariableop_resource:P8
&dense_1_matmul_readvariableop_resource:PP5
'dense_1_biasadd_readvariableop_resource:P
identity’>bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp’=bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp’?bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp’!bidirectional/backward_lstm/while’=bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp’<bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp’>bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp’ bidirectional/forward_lstm/while’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOpV
 bidirectional/forward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:x
.bidirectional/forward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0bidirectional/forward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0bidirectional/forward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ψ
(bidirectional/forward_lstm/strided_sliceStridedSlice)bidirectional/forward_lstm/Shape:output:07bidirectional/forward_lstm/strided_slice/stack:output:09bidirectional/forward_lstm/strided_slice/stack_1:output:09bidirectional/forward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)bidirectional/forward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Δ
'bidirectional/forward_lstm/zeros/packedPack1bidirectional/forward_lstm/strided_slice:output:02bidirectional/forward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&bidirectional/forward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
 bidirectional/forward_lstm/zerosFill0bidirectional/forward_lstm/zeros/packed:output:0/bidirectional/forward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2m
+bidirectional/forward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Θ
)bidirectional/forward_lstm/zeros_1/packedPack1bidirectional/forward_lstm/strided_slice:output:04bidirectional/forward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:m
(bidirectional/forward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Γ
"bidirectional/forward_lstm/zeros_1Fill2bidirectional/forward_lstm/zeros_1/packed:output:01bidirectional/forward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2~
)bidirectional/forward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
$bidirectional/forward_lstm/transpose	Transposeinputs2bidirectional/forward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????z
"bidirectional/forward_lstm/Shape_1Shape(bidirectional/forward_lstm/transpose:y:0*
T0*
_output_shapes
:z
0bidirectional/forward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/forward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2bidirectional/forward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:β
*bidirectional/forward_lstm/strided_slice_1StridedSlice+bidirectional/forward_lstm/Shape_1:output:09bidirectional/forward_lstm/strided_slice_1/stack:output:0;bidirectional/forward_lstm/strided_slice_1/stack_1:output:0;bidirectional/forward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6bidirectional/forward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????
(bidirectional/forward_lstm/TensorArrayV2TensorListReserve?bidirectional/forward_lstm/TensorArrayV2/element_shape:output:03bidirectional/forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?‘
Pbidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ±
Bbidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(bidirectional/forward_lstm/transpose:y:0Ybidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?z
0bidirectional/forward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/forward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2bidirectional/forward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:π
*bidirectional/forward_lstm/strided_slice_2StridedSlice(bidirectional/forward_lstm/transpose:y:09bidirectional/forward_lstm/strided_slice_2/stack:output:0;bidirectional/forward_lstm/strided_slice_2/stack_1:output:0;bidirectional/forward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskΓ
<bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpEbidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0ε
-bidirectional/forward_lstm/lstm_cell_1/MatMulMatMul3bidirectional/forward_lstm/strided_slice_2:output:0Dbidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΗ
>bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGbidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0ί
/bidirectional/forward_lstm/lstm_cell_1/MatMul_1MatMul)bidirectional/forward_lstm/zeros:output:0Fbidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΪ
*bidirectional/forward_lstm/lstm_cell_1/addAddV27bidirectional/forward_lstm/lstm_cell_1/MatMul:product:09bidirectional/forward_lstm/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΑ
=bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpFbidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0γ
.bidirectional/forward_lstm/lstm_cell_1/BiasAddBiasAdd.bidirectional/forward_lstm/lstm_cell_1/add:z:0Ebidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θx
6bidirectional/forward_lstm/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :«
,bidirectional/forward_lstm/lstm_cell_1/splitSplit?bidirectional/forward_lstm/lstm_cell_1/split/split_dim:output:07bidirectional/forward_lstm/lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split’
.bidirectional/forward_lstm/lstm_cell_1/SigmoidSigmoid5bidirectional/forward_lstm/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:?????????2€
0bidirectional/forward_lstm/lstm_cell_1/Sigmoid_1Sigmoid5bidirectional/forward_lstm/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:?????????2Ζ
*bidirectional/forward_lstm/lstm_cell_1/mulMul4bidirectional/forward_lstm/lstm_cell_1/Sigmoid_1:y:0+bidirectional/forward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
+bidirectional/forward_lstm/lstm_cell_1/ReluRelu5bidirectional/forward_lstm/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:?????????2Τ
,bidirectional/forward_lstm/lstm_cell_1/mul_1Mul2bidirectional/forward_lstm/lstm_cell_1/Sigmoid:y:09bidirectional/forward_lstm/lstm_cell_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2Ι
,bidirectional/forward_lstm/lstm_cell_1/add_1AddV2.bidirectional/forward_lstm/lstm_cell_1/mul:z:00bidirectional/forward_lstm/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2€
0bidirectional/forward_lstm/lstm_cell_1/Sigmoid_2Sigmoid5bidirectional/forward_lstm/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????2
-bidirectional/forward_lstm/lstm_cell_1/Relu_1Relu0bidirectional/forward_lstm/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????2Ψ
,bidirectional/forward_lstm/lstm_cell_1/mul_2Mul4bidirectional/forward_lstm/lstm_cell_1/Sigmoid_2:y:0;bidirectional/forward_lstm/lstm_cell_1/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
8bidirectional/forward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   
*bidirectional/forward_lstm/TensorArrayV2_1TensorListReserveAbidirectional/forward_lstm/TensorArrayV2_1/element_shape:output:03bidirectional/forward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?a
bidirectional/forward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3bidirectional/forward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
-bidirectional/forward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : χ
 bidirectional/forward_lstm/whileWhile6bidirectional/forward_lstm/while/loop_counter:output:0<bidirectional/forward_lstm/while/maximum_iterations:output:0(bidirectional/forward_lstm/time:output:03bidirectional/forward_lstm/TensorArrayV2_1:handle:0)bidirectional/forward_lstm/zeros:output:0+bidirectional/forward_lstm/zeros_1:output:03bidirectional/forward_lstm/strided_slice_1:output:0Rbidirectional/forward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ebidirectional_forward_lstm_lstm_cell_1_matmul_readvariableop_resourceGbidirectional_forward_lstm_lstm_cell_1_matmul_1_readvariableop_resourceFbidirectional_forward_lstm_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *7
body/R-
+bidirectional_forward_lstm_while_body_40828*7
cond/R-
+bidirectional_forward_lstm_while_cond_40827*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
Kbidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   
=bidirectional/forward_lstm/TensorArrayV2Stack/TensorListStackTensorListStack)bidirectional/forward_lstm/while:output:3Tbidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0
0bidirectional/forward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
2bidirectional/forward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/forward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
*bidirectional/forward_lstm/strided_slice_3StridedSliceFbidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:09bidirectional/forward_lstm/strided_slice_3/stack:output:0;bidirectional/forward_lstm/strided_slice_3/stack_1:output:0;bidirectional/forward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask
+bidirectional/forward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          η
&bidirectional/forward_lstm/transpose_1	TransposeFbidirectional/forward_lstm/TensorArrayV2Stack/TensorListStack:tensor:04bidirectional/forward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2v
"bidirectional/forward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    W
!bidirectional/backward_lstm/ShapeShapeinputs*
T0*
_output_shapes
:y
/bidirectional/backward_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1bidirectional/backward_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1bidirectional/backward_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:έ
)bidirectional/backward_lstm/strided_sliceStridedSlice*bidirectional/backward_lstm/Shape:output:08bidirectional/backward_lstm/strided_slice/stack:output:0:bidirectional/backward_lstm/strided_slice/stack_1:output:0:bidirectional/backward_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*bidirectional/backward_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Η
(bidirectional/backward_lstm/zeros/packedPack2bidirectional/backward_lstm/strided_slice:output:03bidirectional/backward_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'bidirectional/backward_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ΐ
!bidirectional/backward_lstm/zerosFill1bidirectional/backward_lstm/zeros/packed:output:00bidirectional/backward_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2n
,bidirectional/backward_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2Λ
*bidirectional/backward_lstm/zeros_1/packedPack2bidirectional/backward_lstm/strided_slice:output:05bidirectional/backward_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:n
)bidirectional/backward_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ζ
#bidirectional/backward_lstm/zeros_1Fill3bidirectional/backward_lstm/zeros_1/packed:output:02bidirectional/backward_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
*bidirectional/backward_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ₯
%bidirectional/backward_lstm/transpose	Transposeinputs3bidirectional/backward_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:P?????????|
#bidirectional/backward_lstm/Shape_1Shape)bidirectional/backward_lstm/transpose:y:0*
T0*
_output_shapes
:{
1bidirectional/backward_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3bidirectional/backward_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3bidirectional/backward_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:η
+bidirectional/backward_lstm/strided_slice_1StridedSlice,bidirectional/backward_lstm/Shape_1:output:0:bidirectional/backward_lstm/strided_slice_1/stack:output:0<bidirectional/backward_lstm/strided_slice_1/stack_1:output:0<bidirectional/backward_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7bidirectional/backward_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????
)bidirectional/backward_lstm/TensorArrayV2TensorListReserve@bidirectional/backward_lstm/TensorArrayV2/element_shape:output:04bidirectional/backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?t
*bidirectional/backward_lstm/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Θ
%bidirectional/backward_lstm/ReverseV2	ReverseV2)bidirectional/backward_lstm/transpose:y:03bidirectional/backward_lstm/ReverseV2/axis:output:0*
T0*+
_output_shapes
:P?????????’
Qbidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   Ή
Cbidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor.bidirectional/backward_lstm/ReverseV2:output:0Zbidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?{
1bidirectional/backward_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3bidirectional/backward_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3bidirectional/backward_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:υ
+bidirectional/backward_lstm/strided_slice_2StridedSlice)bidirectional/backward_lstm/transpose:y:0:bidirectional/backward_lstm/strided_slice_2/stack:output:0<bidirectional/backward_lstm/strided_slice_2/stack_1:output:0<bidirectional/backward_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskΕ
=bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpFbidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0θ
.bidirectional/backward_lstm/lstm_cell_2/MatMulMatMul4bidirectional/backward_lstm/strided_slice_2:output:0Ebidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘΙ
?bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpHbidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0β
0bidirectional/backward_lstm/lstm_cell_2/MatMul_1MatMul*bidirectional/backward_lstm/zeros:output:0Gbidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θέ
+bidirectional/backward_lstm/lstm_cell_2/addAddV28bidirectional/backward_lstm/lstm_cell_2/MatMul:product:0:bidirectional/backward_lstm/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:?????????ΘΓ
>bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpGbidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0ζ
/bidirectional/backward_lstm/lstm_cell_2/BiasAddBiasAdd/bidirectional/backward_lstm/lstm_cell_2/add:z:0Fbidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
7bidirectional/backward_lstm/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-bidirectional/backward_lstm/lstm_cell_2/splitSplit@bidirectional/backward_lstm/lstm_cell_2/split/split_dim:output:08bidirectional/backward_lstm/lstm_cell_2/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split€
/bidirectional/backward_lstm/lstm_cell_2/SigmoidSigmoid6bidirectional/backward_lstm/lstm_cell_2/split:output:0*
T0*'
_output_shapes
:?????????2¦
1bidirectional/backward_lstm/lstm_cell_2/Sigmoid_1Sigmoid6bidirectional/backward_lstm/lstm_cell_2/split:output:1*
T0*'
_output_shapes
:?????????2Ι
+bidirectional/backward_lstm/lstm_cell_2/mulMul5bidirectional/backward_lstm/lstm_cell_2/Sigmoid_1:y:0,bidirectional/backward_lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
,bidirectional/backward_lstm/lstm_cell_2/ReluRelu6bidirectional/backward_lstm/lstm_cell_2/split:output:2*
T0*'
_output_shapes
:?????????2Χ
-bidirectional/backward_lstm/lstm_cell_2/mul_1Mul3bidirectional/backward_lstm/lstm_cell_2/Sigmoid:y:0:bidirectional/backward_lstm/lstm_cell_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2Μ
-bidirectional/backward_lstm/lstm_cell_2/add_1AddV2/bidirectional/backward_lstm/lstm_cell_2/mul:z:01bidirectional/backward_lstm/lstm_cell_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2¦
1bidirectional/backward_lstm/lstm_cell_2/Sigmoid_2Sigmoid6bidirectional/backward_lstm/lstm_cell_2/split:output:3*
T0*'
_output_shapes
:?????????2
.bidirectional/backward_lstm/lstm_cell_2/Relu_1Relu1bidirectional/backward_lstm/lstm_cell_2/add_1:z:0*
T0*'
_output_shapes
:?????????2Ϋ
-bidirectional/backward_lstm/lstm_cell_2/mul_2Mul5bidirectional/backward_lstm/lstm_cell_2/Sigmoid_2:y:0<bidirectional/backward_lstm/lstm_cell_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
9bidirectional/backward_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   
+bidirectional/backward_lstm/TensorArrayV2_1TensorListReserveBbidirectional/backward_lstm/TensorArrayV2_1/element_shape:output:04bidirectional/backward_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:ιθ?b
 bidirectional/backward_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4bidirectional/backward_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????p
.bidirectional/backward_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
!bidirectional/backward_lstm/whileWhile7bidirectional/backward_lstm/while/loop_counter:output:0=bidirectional/backward_lstm/while/maximum_iterations:output:0)bidirectional/backward_lstm/time:output:04bidirectional/backward_lstm/TensorArrayV2_1:handle:0*bidirectional/backward_lstm/zeros:output:0,bidirectional/backward_lstm/zeros_1:output:04bidirectional/backward_lstm/strided_slice_1:output:0Sbidirectional/backward_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fbidirectional_backward_lstm_lstm_cell_2_matmul_readvariableop_resourceHbidirectional_backward_lstm_lstm_cell_2_matmul_1_readvariableop_resourceGbidirectional_backward_lstm_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *8
body0R.
,bidirectional_backward_lstm_while_body_40969*8
cond0R.
,bidirectional_backward_lstm_while_cond_40968*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 
Lbidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   
>bidirectional/backward_lstm/TensorArrayV2Stack/TensorListStackTensorListStack*bidirectional/backward_lstm/while:output:3Ubidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:P?????????2*
element_dtype0
1bidirectional/backward_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
3bidirectional/backward_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3bidirectional/backward_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
+bidirectional/backward_lstm/strided_slice_3StridedSliceGbidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:0:bidirectional/backward_lstm/strided_slice_3/stack:output:0<bidirectional/backward_lstm/strided_slice_3/stack_1:output:0<bidirectional/backward_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask
,bidirectional/backward_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          κ
'bidirectional/backward_lstm/transpose_1	TransposeGbidirectional/backward_lstm/TensorArrayV2Stack/TensorListStack:tensor:05bidirectional/backward_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P2w
#bidirectional/backward_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    [
bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :κ
bidirectional/concatConcatV23bidirectional/forward_lstm/strided_slice_3:output:04bidirectional/backward_lstm/strided_slice_3:output:0"bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype0
dense/MatMulMatMulbidirectional/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pa
activation/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????PZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout/MulMulactivation/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????Pb
dropout/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:­
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed2????c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>Ύ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pg
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????P
NoOpNoOp?^bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp>^bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp@^bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp"^bidirectional/backward_lstm/while>^bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp=^bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp?^bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp!^bidirectional/forward_lstm/while^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : 2
>bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp>bidirectional/backward_lstm/lstm_cell_2/BiasAdd/ReadVariableOp2~
=bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp=bidirectional/backward_lstm/lstm_cell_2/MatMul/ReadVariableOp2
?bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp?bidirectional/backward_lstm/lstm_cell_2/MatMul_1/ReadVariableOp2F
!bidirectional/backward_lstm/while!bidirectional/backward_lstm/while2~
=bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp=bidirectional/forward_lstm/lstm_cell_1/BiasAdd/ReadVariableOp2|
<bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp<bidirectional/forward_lstm/lstm_cell_1/MatMul/ReadVariableOp2
>bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp>bidirectional/forward_lstm/lstm_cell_1/MatMul_1/ReadVariableOp2D
 bidirectional/forward_lstm/while bidirectional/forward_lstm/while2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
Υ
`
B__inference_dropout_layer_call_and_return_conditional_losses_42359

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????P[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
Ή
Φ
backward_lstm_while_cond_397118
4backward_lstm_while_backward_lstm_while_loop_counter>
:backward_lstm_while_backward_lstm_while_maximum_iterations#
backward_lstm_while_placeholder%
!backward_lstm_while_placeholder_1%
!backward_lstm_while_placeholder_2%
!backward_lstm_while_placeholder_3:
6backward_lstm_while_less_backward_lstm_strided_slice_1O
Kbackward_lstm_while_backward_lstm_while_cond_39711___redundant_placeholder0O
Kbackward_lstm_while_backward_lstm_while_cond_39711___redundant_placeholder1O
Kbackward_lstm_while_backward_lstm_while_cond_39711___redundant_placeholder2O
Kbackward_lstm_while_backward_lstm_while_cond_39711___redundant_placeholder3 
backward_lstm_while_identity

backward_lstm/while/LessLessbackward_lstm_while_placeholder6backward_lstm_while_less_backward_lstm_strided_slice_1*
T0*
_output_shapes
: g
backward_lstm/while/IdentityIdentitybackward_lstm/while/Less:z:0*
T0
*
_output_shapes
: "E
backward_lstm_while_identity%backward_lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
°
Ύ
while_cond_39210
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_39210___redundant_placeholder03
/while_while_cond_39210___redundant_placeholder13
/while_while_cond_39210___redundant_placeholder23
/while_while_cond_39210___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
σ

H__inference_bidirectional_layer_call_and_return_conditional_losses_39491

inputs%
forward_lstm_39474:	Θ%
forward_lstm_39476:	2Θ!
forward_lstm_39478:	Θ&
backward_lstm_39481:	Θ&
backward_lstm_39483:	2Θ"
backward_lstm_39485:	Θ
identity’%backward_lstm/StatefulPartitionedCall’$forward_lstm/StatefulPartitionedCall
$forward_lstm/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_39474forward_lstm_39476forward_lstm_39478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_forward_lstm_layer_call_and_return_conditional_losses_39460
%backward_lstm/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_39481backward_lstm_39483backward_lstm_39485*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_39295M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Β
concatConcatV2-forward_lstm/StatefulPartitionedCall:output:0.backward_lstm/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????d^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????d
NoOpNoOp&^backward_lstm/StatefulPartitionedCall%^forward_lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2N
%backward_lstm/StatefulPartitionedCall%backward_lstm/StatefulPartitionedCall2L
$forward_lstm/StatefulPartitionedCall$forward_lstm/StatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ϋ

F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_43826

inputs
states_0
states_11
matmul_readvariableop_resource:	Θ3
 matmul_1_readvariableop_resource:	2Θ.
biasadd_readvariableop_resource:	Θ
identity

identity_1

identity_2’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp’MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Θ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Θ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Θe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:?????????Θs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Θ*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ΘQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ά
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2N
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????2Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????:?????????2:?????????2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ζ
serving_default²
W
bidirectional_input@
%serving_default_bidirectional_input:0?????????P;
dense_10
StatefulPartitionedCall:0?????????Ptensorflow/serving/predict:ϋΰ
υ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Μ
forward_layer
backward_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
Ό
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)_random_generator
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
»

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer

4iter

5beta_1

6beta_2
	7decay
8learning_ratemm,m-m9m:m ;m‘<m’=m£>m€v₯v¦,v§-v¨9v©:vͺ;v«<v¬=v­>v?"
	optimizer
f
90
:1
;2
<3
=4
>5
6
7
,8
-9"
trackable_list_wrapper
f
90
:1
;2
<3
=4
>5
6
7
,8
-9"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
φ2σ
*__inference_sequential_layer_call_fn_39882
*__inference_sequential_layer_call_fn_40444
*__inference_sequential_layer_call_fn_40469
*__inference_sequential_layer_call_fn_40355ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
E__inference_sequential_layer_call_and_return_conditional_losses_40769
E__inference_sequential_layer_call_and_return_conditional_losses_41076
E__inference_sequential_layer_call_and_return_conditional_losses_40384
E__inference_sequential_layer_call_and_return_conditional_losses_40413ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΧBΤ
 __inference__wrapped_model_38106bidirectional_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
Dserving_default"
signature_map
Ϊ
Ecell
F
state_spec
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K_random_generator
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Ϊ
Ncell
O
state_spec
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T_random_generator
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
J
90
:1
;2
<3
=4
>5"
trackable_list_wrapper
J
90
:1
;2
<3
=4
>5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¨2₯
-__inference_bidirectional_layer_call_fn_41120
-__inference_bidirectional_layer_call_fn_41137
-__inference_bidirectional_layer_call_fn_41154
-__inference_bidirectional_layer_call_fn_41171ζ
έ²Ω
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
H__inference_bidirectional_layer_call_and_return_conditional_losses_41457
H__inference_bidirectional_layer_call_and_return_conditional_losses_41743
H__inference_bidirectional_layer_call_and_return_conditional_losses_42029
H__inference_bidirectional_layer_call_and_return_conditional_losses_42315ζ
έ²Ω
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
:dP2dense/kernel
:P2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ο2Μ
%__inference_dense_layer_call_fn_42324’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_dense_layer_call_and_return_conditional_losses_42334’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
Τ2Ρ
*__inference_activation_layer_call_fn_42339’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_activation_layer_call_and_return_conditional_losses_42344’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
%	variables
&trainable_variables
'regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
'__inference_dropout_layer_call_fn_42349
'__inference_dropout_layer_call_fn_42354΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Β2Ώ
B__inference_dropout_layer_call_and_return_conditional_losses_42359
B__inference_dropout_layer_call_and_return_conditional_losses_42371΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 :PP2dense_1/kernel
:P2dense_1/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ρ2Ξ
'__inference_dense_1_layer_call_fn_42380’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_dense_1_layer_call_and_return_conditional_losses_42390’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
@:>	Θ2-bidirectional/forward_lstm/lstm_cell_1/kernel
J:H	2Θ27bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel
::8Θ2+bidirectional/forward_lstm/lstm_cell_1/bias
A:?	Θ2.bidirectional/backward_lstm/lstm_cell_2/kernel
K:I	2Θ28bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel
;:9Θ2,bidirectional/backward_lstm/lstm_cell_2/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΦBΣ
#__inference_signature_wrapper_41103bidirectional_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψ
q
state_size

9kernel
:recurrent_kernel
;bias
r	variables
strainable_variables
tregularization_losses
u	keras_api
v_random_generator
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ή

ystates
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
,__inference_forward_lstm_layer_call_fn_42401
,__inference_forward_lstm_layer_call_fn_42412
,__inference_forward_lstm_layer_call_fn_42423
,__inference_forward_lstm_layer_call_fn_42434Υ
Μ²Θ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
?2ό
G__inference_forward_lstm_layer_call_and_return_conditional_losses_42577
G__inference_forward_lstm_layer_call_and_return_conditional_losses_42720
G__inference_forward_lstm_layer_call_and_return_conditional_losses_42863
G__inference_forward_lstm_layer_call_and_return_conditional_losses_43006Υ
Μ²Θ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
?

state_size

<kernel
=recurrent_kernel
>bias
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ώ
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
-__inference_backward_lstm_layer_call_fn_43017
-__inference_backward_lstm_layer_call_fn_43028
-__inference_backward_lstm_layer_call_fn_43039
-__inference_backward_lstm_layer_call_fn_43050Υ
Μ²Θ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
H__inference_backward_lstm_layer_call_and_return_conditional_losses_43195
H__inference_backward_lstm_layer_call_and_return_conditional_losses_43340
H__inference_backward_lstm_layer_call_and_return_conditional_losses_43485
H__inference_backward_lstm_layer_call_and_return_conditional_losses_43630Υ
Μ²Θ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
r	variables
strainable_variables
tregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_lstm_cell_1_layer_call_fn_43647
+__inference_lstm_cell_1_layer_call_fn_43664Ύ
΅²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_43696
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_43728Ύ
΅²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_lstm_cell_2_layer_call_fn_43745
+__inference_lstm_cell_2_layer_call_fn_43762Ύ
΅²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_43794
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_43826Ύ
΅²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
#:!dP2Adam/dense/kernel/m
:P2Adam/dense/bias/m
%:#PP2Adam/dense_1/kernel/m
:P2Adam/dense_1/bias/m
E:C	Θ24Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/m
O:M	2Θ2>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/m
?:=Θ22Adam/bidirectional/forward_lstm/lstm_cell_1/bias/m
F:D	Θ25Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/m
P:N	2Θ2?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/m
@:>Θ23Adam/bidirectional/backward_lstm/lstm_cell_2/bias/m
#:!dP2Adam/dense/kernel/v
:P2Adam/dense/bias/v
%:#PP2Adam/dense_1/kernel/v
:P2Adam/dense_1/bias/v
E:C	Θ24Adam/bidirectional/forward_lstm/lstm_cell_1/kernel/v
O:M	2Θ2>Adam/bidirectional/forward_lstm/lstm_cell_1/recurrent_kernel/v
?:=Θ22Adam/bidirectional/forward_lstm/lstm_cell_1/bias/v
F:D	Θ25Adam/bidirectional/backward_lstm/lstm_cell_2/kernel/v
P:N	2Θ2?Adam/bidirectional/backward_lstm/lstm_cell_2/recurrent_kernel/v
@:>Θ23Adam/bidirectional/backward_lstm/lstm_cell_2/bias/v¦
 __inference__wrapped_model_38106
9:;<=>,-@’=
6’3
1.
bidirectional_input?????????P
ͺ "1ͺ.
,
dense_1!
dense_1?????????P‘
E__inference_activation_layer_call_and_return_conditional_losses_42344X/’,
%’"
 
inputs?????????P
ͺ "%’"

0?????????P
 y
*__inference_activation_layer_call_fn_42339K/’,
%’"
 
inputs?????????P
ͺ "?????????PΙ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_43195}<=>O’L
E’B
41
/,
inputs/0??????????????????

 
p 

 
ͺ "%’"

0?????????2
 Ι
H__inference_backward_lstm_layer_call_and_return_conditional_losses_43340}<=>O’L
E’B
41
/,
inputs/0??????????????????

 
p

 
ͺ "%’"

0?????????2
 Λ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_43485<=>Q’N
G’D
63
inputs'???????????????????????????

 
p 

 
ͺ "%’"

0?????????2
 Λ
H__inference_backward_lstm_layer_call_and_return_conditional_losses_43630<=>Q’N
G’D
63
inputs'???????????????????????????

 
p

 
ͺ "%’"

0?????????2
 ‘
-__inference_backward_lstm_layer_call_fn_43017p<=>O’L
E’B
41
/,
inputs/0??????????????????

 
p 

 
ͺ "?????????2‘
-__inference_backward_lstm_layer_call_fn_43028p<=>O’L
E’B
41
/,
inputs/0??????????????????

 
p

 
ͺ "?????????2£
-__inference_backward_lstm_layer_call_fn_43039r<=>Q’N
G’D
63
inputs'???????????????????????????

 
p 

 
ͺ "?????????2£
-__inference_backward_lstm_layer_call_fn_43050r<=>Q’N
G’D
63
inputs'???????????????????????????

 
p

 
ͺ "?????????2Ϊ
H__inference_bidirectional_layer_call_and_return_conditional_losses_414579:;<=>\’Y
R’O
=:
85
inputs/0'???????????????????????????
p 

 

 

 
ͺ "%’"

0?????????d
 Ϊ
H__inference_bidirectional_layer_call_and_return_conditional_losses_417439:;<=>\’Y
R’O
=:
85
inputs/0'???????????????????????????
p

 

 

 
ͺ "%’"

0?????????d
 ΐ
H__inference_bidirectional_layer_call_and_return_conditional_losses_42029t9:;<=>C’@
9’6
$!
inputs?????????P
p 

 

 

 
ͺ "%’"

0?????????d
 ΐ
H__inference_bidirectional_layer_call_and_return_conditional_losses_42315t9:;<=>C’@
9’6
$!
inputs?????????P
p

 

 

 
ͺ "%’"

0?????????d
 ²
-__inference_bidirectional_layer_call_fn_411209:;<=>\’Y
R’O
=:
85
inputs/0'???????????????????????????
p 

 

 

 
ͺ "?????????d²
-__inference_bidirectional_layer_call_fn_411379:;<=>\’Y
R’O
=:
85
inputs/0'???????????????????????????
p

 

 

 
ͺ "?????????d
-__inference_bidirectional_layer_call_fn_41154g9:;<=>C’@
9’6
$!
inputs?????????P
p 

 

 

 
ͺ "?????????d
-__inference_bidirectional_layer_call_fn_41171g9:;<=>C’@
9’6
$!
inputs?????????P
p

 

 

 
ͺ "?????????d’
B__inference_dense_1_layer_call_and_return_conditional_losses_42390\,-/’,
%’"
 
inputs?????????P
ͺ "%’"

0?????????P
 z
'__inference_dense_1_layer_call_fn_42380O,-/’,
%’"
 
inputs?????????P
ͺ "?????????P 
@__inference_dense_layer_call_and_return_conditional_losses_42334\/’,
%’"
 
inputs?????????d
ͺ "%’"

0?????????P
 x
%__inference_dense_layer_call_fn_42324O/’,
%’"
 
inputs?????????d
ͺ "?????????P’
B__inference_dropout_layer_call_and_return_conditional_losses_42359\3’0
)’&
 
inputs?????????P
p 
ͺ "%’"

0?????????P
 ’
B__inference_dropout_layer_call_and_return_conditional_losses_42371\3’0
)’&
 
inputs?????????P
p
ͺ "%’"

0?????????P
 z
'__inference_dropout_layer_call_fn_42349O3’0
)’&
 
inputs?????????P
p 
ͺ "?????????Pz
'__inference_dropout_layer_call_fn_42354O3’0
)’&
 
inputs?????????P
p
ͺ "?????????PΘ
G__inference_forward_lstm_layer_call_and_return_conditional_losses_42577}9:;O’L
E’B
41
/,
inputs/0??????????????????

 
p 

 
ͺ "%’"

0?????????2
 Θ
G__inference_forward_lstm_layer_call_and_return_conditional_losses_42720}9:;O’L
E’B
41
/,
inputs/0??????????????????

 
p

 
ͺ "%’"

0?????????2
 Κ
G__inference_forward_lstm_layer_call_and_return_conditional_losses_428639:;Q’N
G’D
63
inputs'???????????????????????????

 
p 

 
ͺ "%’"

0?????????2
 Κ
G__inference_forward_lstm_layer_call_and_return_conditional_losses_430069:;Q’N
G’D
63
inputs'???????????????????????????

 
p

 
ͺ "%’"

0?????????2
  
,__inference_forward_lstm_layer_call_fn_42401p9:;O’L
E’B
41
/,
inputs/0??????????????????

 
p 

 
ͺ "?????????2 
,__inference_forward_lstm_layer_call_fn_42412p9:;O’L
E’B
41
/,
inputs/0??????????????????

 
p

 
ͺ "?????????2’
,__inference_forward_lstm_layer_call_fn_42423r9:;Q’N
G’D
63
inputs'???????????????????????????

 
p 

 
ͺ "?????????2’
,__inference_forward_lstm_layer_call_fn_42434r9:;Q’N
G’D
63
inputs'???????????????????????????

 
p

 
ͺ "?????????2Θ
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_43696ύ9:;’}
v’s
 
inputs?????????
K’H
"
states/0?????????2
"
states/1?????????2
p 
ͺ "s’p
i’f

0/0?????????2
EB

0/1/0?????????2

0/1/1?????????2
 Θ
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_43728ύ9:;’}
v’s
 
inputs?????????
K’H
"
states/0?????????2
"
states/1?????????2
p
ͺ "s’p
i’f

0/0?????????2
EB

0/1/0?????????2

0/1/1?????????2
 
+__inference_lstm_cell_1_layer_call_fn_43647ν9:;’}
v’s
 
inputs?????????
K’H
"
states/0?????????2
"
states/1?????????2
p 
ͺ "c’`

0?????????2
A>

1/0?????????2

1/1?????????2
+__inference_lstm_cell_1_layer_call_fn_43664ν9:;’}
v’s
 
inputs?????????
K’H
"
states/0?????????2
"
states/1?????????2
p
ͺ "c’`

0?????????2
A>

1/0?????????2

1/1?????????2Θ
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_43794ύ<=>’}
v’s
 
inputs?????????
K’H
"
states/0?????????2
"
states/1?????????2
p 
ͺ "s’p
i’f

0/0?????????2
EB

0/1/0?????????2

0/1/1?????????2
 Θ
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_43826ύ<=>’}
v’s
 
inputs?????????
K’H
"
states/0?????????2
"
states/1?????????2
p
ͺ "s’p
i’f

0/0?????????2
EB

0/1/0?????????2

0/1/1?????????2
 
+__inference_lstm_cell_2_layer_call_fn_43745ν<=>’}
v’s
 
inputs?????????
K’H
"
states/0?????????2
"
states/1?????????2
p 
ͺ "c’`

0?????????2
A>

1/0?????????2

1/1?????????2
+__inference_lstm_cell_2_layer_call_fn_43762ν<=>’}
v’s
 
inputs?????????
K’H
"
states/0?????????2
"
states/1?????????2
p
ͺ "c’`

0?????????2
A>

1/0?????????2

1/1?????????2Ζ
E__inference_sequential_layer_call_and_return_conditional_losses_40384}
9:;<=>,-H’E
>’;
1.
bidirectional_input?????????P
p 

 
ͺ "%’"

0?????????P
 Ζ
E__inference_sequential_layer_call_and_return_conditional_losses_40413}
9:;<=>,-H’E
>’;
1.
bidirectional_input?????????P
p

 
ͺ "%’"

0?????????P
 Ή
E__inference_sequential_layer_call_and_return_conditional_losses_40769p
9:;<=>,-;’8
1’.
$!
inputs?????????P
p 

 
ͺ "%’"

0?????????P
 Ή
E__inference_sequential_layer_call_and_return_conditional_losses_41076p
9:;<=>,-;’8
1’.
$!
inputs?????????P
p

 
ͺ "%’"

0?????????P
 
*__inference_sequential_layer_call_fn_39882p
9:;<=>,-H’E
>’;
1.
bidirectional_input?????????P
p 

 
ͺ "?????????P
*__inference_sequential_layer_call_fn_40355p
9:;<=>,-H’E
>’;
1.
bidirectional_input?????????P
p

 
ͺ "?????????P
*__inference_sequential_layer_call_fn_40444c
9:;<=>,-;’8
1’.
$!
inputs?????????P
p 

 
ͺ "?????????P
*__inference_sequential_layer_call_fn_40469c
9:;<=>,-;’8
1’.
$!
inputs?????????P
p

 
ͺ "?????????Pΐ
#__inference_signature_wrapper_41103
9:;<=>,-W’T
’ 
MͺJ
H
bidirectional_input1.
bidirectional_input?????????P"1ͺ.
,
dense_1!
dense_1?????????P