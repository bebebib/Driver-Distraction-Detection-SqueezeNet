๗1
อฃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
พ
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ดล'

Conv2D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameConv2D_1/kernel
{
#Conv2D_1/kernel/Read/ReadVariableOpReadVariableOpConv2D_1/kernel*&
_output_shapes
:@*
dtype0
r
Conv2D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv2D_1/bias
k
!Conv2D_1/bias/Read/ReadVariableOpReadVariableOpConv2D_1/bias*
_output_shapes
:@*
dtype0

SqueezeFire2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameSqueezeFire2/kernel

'SqueezeFire2/kernel/Read/ReadVariableOpReadVariableOpSqueezeFire2/kernel*&
_output_shapes
:@*
dtype0
z
SqueezeFire2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameSqueezeFire2/bias
s
%SqueezeFire2/bias/Read/ReadVariableOpReadVariableOpSqueezeFire2/bias*
_output_shapes
:*
dtype0

Expand1x1Fire2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameExpand1x1Fire2/kernel

)Expand1x1Fire2/kernel/Read/ReadVariableOpReadVariableOpExpand1x1Fire2/kernel*&
_output_shapes
:@*
dtype0
~
Expand1x1Fire2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameExpand1x1Fire2/bias
w
'Expand1x1Fire2/bias/Read/ReadVariableOpReadVariableOpExpand1x1Fire2/bias*
_output_shapes
:@*
dtype0

Expand3x3Fire2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameExpand3x3Fire2/kernel

)Expand3x3Fire2/kernel/Read/ReadVariableOpReadVariableOpExpand3x3Fire2/kernel*&
_output_shapes
:@*
dtype0
~
Expand3x3Fire2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameExpand3x3Fire2/bias
w
'Expand3x3Fire2/bias/Read/ReadVariableOpReadVariableOpExpand3x3Fire2/bias*
_output_shapes
:@*
dtype0

SqueezeFire3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameSqueezeFire3/kernel

'SqueezeFire3/kernel/Read/ReadVariableOpReadVariableOpSqueezeFire3/kernel*'
_output_shapes
:*
dtype0
z
SqueezeFire3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameSqueezeFire3/bias
s
%SqueezeFire3/bias/Read/ReadVariableOpReadVariableOpSqueezeFire3/bias*
_output_shapes
:*
dtype0

Expand1x1Fire3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameExpand1x1Fire3/kernel

)Expand1x1Fire3/kernel/Read/ReadVariableOpReadVariableOpExpand1x1Fire3/kernel*&
_output_shapes
:@*
dtype0
~
Expand1x1Fire3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameExpand1x1Fire3/bias
w
'Expand1x1Fire3/bias/Read/ReadVariableOpReadVariableOpExpand1x1Fire3/bias*
_output_shapes
:@*
dtype0

Expand3x3Fire3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameExpand3x3Fire3/kernel

)Expand3x3Fire3/kernel/Read/ReadVariableOpReadVariableOpExpand3x3Fire3/kernel*&
_output_shapes
:@*
dtype0
~
Expand3x3Fire3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameExpand3x3Fire3/bias
w
'Expand3x3Fire3/bias/Read/ReadVariableOpReadVariableOpExpand3x3Fire3/bias*
_output_shapes
:@*
dtype0

SqueezeFire4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameSqueezeFire4/kernel

'SqueezeFire4/kernel/Read/ReadVariableOpReadVariableOpSqueezeFire4/kernel*'
_output_shapes
: *
dtype0
z
SqueezeFire4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSqueezeFire4/bias
s
%SqueezeFire4/bias/Read/ReadVariableOpReadVariableOpSqueezeFire4/bias*
_output_shapes
: *
dtype0

Expand1x1Fire4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameExpand1x1Fire4/kernel

)Expand1x1Fire4/kernel/Read/ReadVariableOpReadVariableOpExpand1x1Fire4/kernel*'
_output_shapes
: *
dtype0

Expand1x1Fire4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameExpand1x1Fire4/bias
x
'Expand1x1Fire4/bias/Read/ReadVariableOpReadVariableOpExpand1x1Fire4/bias*
_output_shapes	
:*
dtype0

Expand3x3Fire4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameExpand3x3Fire4/kernel

)Expand3x3Fire4/kernel/Read/ReadVariableOpReadVariableOpExpand3x3Fire4/kernel*'
_output_shapes
: *
dtype0

Expand3x3Fire4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameExpand3x3Fire4/bias
x
'Expand3x3Fire4/bias/Read/ReadVariableOpReadVariableOpExpand3x3Fire4/bias*
_output_shapes	
:*
dtype0

SqueezeFire5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameSqueezeFire5/kernel

'SqueezeFire5/kernel/Read/ReadVariableOpReadVariableOpSqueezeFire5/kernel*'
_output_shapes
: *
dtype0
z
SqueezeFire5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSqueezeFire5/bias
s
%SqueezeFire5/bias/Read/ReadVariableOpReadVariableOpSqueezeFire5/bias*
_output_shapes
: *
dtype0

Expand1x1Fire5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameExpand1x1Fire5/kernel

)Expand1x1Fire5/kernel/Read/ReadVariableOpReadVariableOpExpand1x1Fire5/kernel*'
_output_shapes
: *
dtype0

Expand1x1Fire5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameExpand1x1Fire5/bias
x
'Expand1x1Fire5/bias/Read/ReadVariableOpReadVariableOpExpand1x1Fire5/bias*
_output_shapes	
:*
dtype0

Expand3x3Fire5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameExpand3x3Fire5/kernel

)Expand3x3Fire5/kernel/Read/ReadVariableOpReadVariableOpExpand3x3Fire5/kernel*'
_output_shapes
: *
dtype0

Expand3x3Fire5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameExpand3x3Fire5/bias
x
'Expand3x3Fire5/bias/Read/ReadVariableOpReadVariableOpExpand3x3Fire5/bias*
_output_shapes	
:*
dtype0

SqueezeFire6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameSqueezeFire6/kernel

'SqueezeFire6/kernel/Read/ReadVariableOpReadVariableOpSqueezeFire6/kernel*'
_output_shapes
:0*
dtype0
z
SqueezeFire6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameSqueezeFire6/bias
s
%SqueezeFire6/bias/Read/ReadVariableOpReadVariableOpSqueezeFire6/bias*
_output_shapes
:0*
dtype0

Expand1x1Fire6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0ภ*&
shared_nameExpand1x1Fire6/kernel

)Expand1x1Fire6/kernel/Read/ReadVariableOpReadVariableOpExpand1x1Fire6/kernel*'
_output_shapes
:0ภ*
dtype0

Expand1x1Fire6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ภ*$
shared_nameExpand1x1Fire6/bias
x
'Expand1x1Fire6/bias/Read/ReadVariableOpReadVariableOpExpand1x1Fire6/bias*
_output_shapes	
:ภ*
dtype0

Expand3x3Fire6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0ภ*&
shared_nameExpand3x3Fire6/kernel

)Expand3x3Fire6/kernel/Read/ReadVariableOpReadVariableOpExpand3x3Fire6/kernel*'
_output_shapes
:0ภ*
dtype0

Expand3x3Fire6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ภ*$
shared_nameExpand3x3Fire6/bias
x
'Expand3x3Fire6/bias/Read/ReadVariableOpReadVariableOpExpand3x3Fire6/bias*
_output_shapes	
:ภ*
dtype0

SqueezeFire7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameSqueezeFire7/kernel

'SqueezeFire7/kernel/Read/ReadVariableOpReadVariableOpSqueezeFire7/kernel*'
_output_shapes
:0*
dtype0
z
SqueezeFire7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameSqueezeFire7/bias
s
%SqueezeFire7/bias/Read/ReadVariableOpReadVariableOpSqueezeFire7/bias*
_output_shapes
:0*
dtype0

Expand1x1Fire7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0ภ*&
shared_nameExpand1x1Fire7/kernel

)Expand1x1Fire7/kernel/Read/ReadVariableOpReadVariableOpExpand1x1Fire7/kernel*'
_output_shapes
:0ภ*
dtype0

Expand1x1Fire7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ภ*$
shared_nameExpand1x1Fire7/bias
x
'Expand1x1Fire7/bias/Read/ReadVariableOpReadVariableOpExpand1x1Fire7/bias*
_output_shapes	
:ภ*
dtype0

Expand3x3Fire7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0ภ*&
shared_nameExpand3x3Fire7/kernel

)Expand3x3Fire7/kernel/Read/ReadVariableOpReadVariableOpExpand3x3Fire7/kernel*'
_output_shapes
:0ภ*
dtype0

Expand3x3Fire7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ภ*$
shared_nameExpand3x3Fire7/bias
x
'Expand3x3Fire7/bias/Read/ReadVariableOpReadVariableOpExpand3x3Fire7/bias*
_output_shapes	
:ภ*
dtype0

SqueezeFire8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameSqueezeFire8/kernel

'SqueezeFire8/kernel/Read/ReadVariableOpReadVariableOpSqueezeFire8/kernel*'
_output_shapes
:@*
dtype0
z
SqueezeFire8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameSqueezeFire8/bias
s
%SqueezeFire8/bias/Read/ReadVariableOpReadVariableOpSqueezeFire8/bias*
_output_shapes
:@*
dtype0

Expand1x1Fire8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameExpand1x1Fire8/kernel

)Expand1x1Fire8/kernel/Read/ReadVariableOpReadVariableOpExpand1x1Fire8/kernel*'
_output_shapes
:@*
dtype0

Expand1x1Fire8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameExpand1x1Fire8/bias
x
'Expand1x1Fire8/bias/Read/ReadVariableOpReadVariableOpExpand1x1Fire8/bias*
_output_shapes	
:*
dtype0

Expand3x3Fire8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameExpand3x3Fire8/kernel

)Expand3x3Fire8/kernel/Read/ReadVariableOpReadVariableOpExpand3x3Fire8/kernel*'
_output_shapes
:@*
dtype0

Expand3x3Fire8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameExpand3x3Fire8/bias
x
'Expand3x3Fire8/bias/Read/ReadVariableOpReadVariableOpExpand3x3Fire8/bias*
_output_shapes	
:*
dtype0

SqueezeFire9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameSqueezeFire9/kernel

'SqueezeFire9/kernel/Read/ReadVariableOpReadVariableOpSqueezeFire9/kernel*'
_output_shapes
:@*
dtype0
z
SqueezeFire9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameSqueezeFire9/bias
s
%SqueezeFire9/bias/Read/ReadVariableOpReadVariableOpSqueezeFire9/bias*
_output_shapes
:@*
dtype0

Expand1x1Fire9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameExpand1x1Fire9/kernel

)Expand1x1Fire9/kernel/Read/ReadVariableOpReadVariableOpExpand1x1Fire9/kernel*'
_output_shapes
:@*
dtype0

Expand1x1Fire9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameExpand1x1Fire9/bias
x
'Expand1x1Fire9/bias/Read/ReadVariableOpReadVariableOpExpand1x1Fire9/bias*
_output_shapes	
:*
dtype0

Expand3x3Fire9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameExpand3x3Fire9/kernel

)Expand3x3Fire9/kernel/Read/ReadVariableOpReadVariableOpExpand3x3Fire9/kernel*'
_output_shapes
:@*
dtype0

Expand3x3Fire9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameExpand3x3Fire9/bias
x
'Expand3x3Fire9/bias/Read/ReadVariableOpReadVariableOpExpand3x3Fire9/bias*
_output_shapes	
:*
dtype0

DenseFinal/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ค*"
shared_nameDenseFinal/kernel
y
%DenseFinal/kernel/Read/ReadVariableOpReadVariableOpDenseFinal/kernel* 
_output_shapes
:
ค*
dtype0
v
DenseFinal/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameDenseFinal/bias
o
#DenseFinal/bias/Read/ReadVariableOpReadVariableOpDenseFinal/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
ีณ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ณ
valueณBณ B๘ฒ
๐

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer_with_weights-12
layer-18
layer-19
layer_with_weights-13
layer-20
layer_with_weights-14
layer-21
layer_with_weights-15
layer-22
layer-23
layer_with_weights-16
layer-24
layer_with_weights-17
layer-25
layer_with_weights-18
layer-26
layer-27
layer_with_weights-19
layer-28
layer_with_weights-20
layer-29
layer_with_weights-21
layer-30
 layer-31
!layer-32
"layer_with_weights-22
"layer-33
#layer_with_weights-23
#layer-34
$layer_with_weights-24
$layer-35
%layer-36
&layer-37
'layer-38
(layer_with_weights-25
(layer-39
#)_self_saveable_object_factories
*	optimizer
+
signatures
,regularization_losses
-	variables
.trainable_variables
/	keras_api
%
#0_self_saveable_object_factories


1kernel
2bias
#3_self_saveable_object_factories
4regularization_losses
5	variables
6trainable_variables
7	keras_api
w
#8_self_saveable_object_factories
9regularization_losses
:	variables
;trainable_variables
<	keras_api


=kernel
>bias
#?_self_saveable_object_factories
@regularization_losses
A	variables
Btrainable_variables
C	keras_api


Dkernel
Ebias
#F_self_saveable_object_factories
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api


Kkernel
Lbias
#M_self_saveable_object_factories
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
w
#R_self_saveable_object_factories
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api


Wkernel
Xbias
#Y_self_saveable_object_factories
Zregularization_losses
[	variables
\trainable_variables
]	keras_api


^kernel
_bias
#`_self_saveable_object_factories
aregularization_losses
b	variables
ctrainable_variables
d	keras_api


ekernel
fbias
#g_self_saveable_object_factories
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
w
#l_self_saveable_object_factories
mregularization_losses
n	variables
otrainable_variables
p	keras_api


qkernel
rbias
#s_self_saveable_object_factories
tregularization_losses
u	variables
vtrainable_variables
w	keras_api


xkernel
ybias
#z_self_saveable_object_factories
{regularization_losses
|	variables
}trainable_variables
~	keras_api


kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$?_self_saveable_object_factories
กregularization_losses
ข	variables
ฃtrainable_variables
ค	keras_api
|
$ฅ_self_saveable_object_factories
ฆregularization_losses
ง	variables
จtrainable_variables
ฉ	keras_api

ชkernel
	ซbias
$ฌ_self_saveable_object_factories
ญregularization_losses
ฎ	variables
ฏtrainable_variables
ฐ	keras_api

ฑkernel
	ฒbias
$ณ_self_saveable_object_factories
ดregularization_losses
ต	variables
ถtrainable_variables
ท	keras_api

ธkernel
	นbias
$บ_self_saveable_object_factories
ปregularization_losses
ผ	variables
ฝtrainable_variables
พ	keras_api
|
$ฟ_self_saveable_object_factories
ภregularization_losses
ม	variables
ยtrainable_variables
ร	keras_api

ฤkernel
	ลbias
$ฦ_self_saveable_object_factories
วregularization_losses
ศ	variables
ษtrainable_variables
ส	keras_api

หkernel
	ฬbias
$อ_self_saveable_object_factories
ฮregularization_losses
ฯ	variables
ะtrainable_variables
ั	keras_api

าkernel
	ำbias
$ิ_self_saveable_object_factories
ีregularization_losses
ึ	variables
ืtrainable_variables
ุ	keras_api
|
$ู_self_saveable_object_factories
ฺregularization_losses
?	variables
?trainable_variables
?	keras_api

?kernel
	฿bias
$เ_self_saveable_object_factories
แregularization_losses
โ	variables
ใtrainable_variables
ไ	keras_api

ๅkernel
	ๆbias
$็_self_saveable_object_factories
่regularization_losses
้	variables
๊trainable_variables
๋	keras_api

์kernel
	ํbias
$๎_self_saveable_object_factories
๏regularization_losses
๐	variables
๑trainable_variables
๒	keras_api
|
$๓_self_saveable_object_factories
๔regularization_losses
๕	variables
๖trainable_variables
๗	keras_api
|
$๘_self_saveable_object_factories
๙regularization_losses
๚	variables
๛trainable_variables
?	keras_api

?kernel
	?bias
$?_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
?	keras_api

กkernel
	ขbias
$ฃ_self_saveable_object_factories
คregularization_losses
ฅ	variables
ฆtrainable_variables
ง	keras_api
 
:
	จiter

ฉdecay
ชlearning_rate
ซmomentum
 
 
ท
10
21
=2
>3
D4
E5
K6
L7
W8
X9
^10
_11
e12
f13
q14
r15
x16
y17
18
19
20
21
22
23
24
25
ช26
ซ27
ฑ28
ฒ29
ธ30
น31
ฤ32
ล33
ห34
ฬ35
า36
ำ37
?38
฿39
ๅ40
ๆ41
์42
ํ43
?44
?45
46
47
48
49
ก50
ข51
ท
10
21
=2
>3
D4
E5
K6
L7
W8
X9
^10
_11
e12
f13
q14
r15
x16
y17
18
19
20
21
22
23
24
25
ช26
ซ27
ฑ28
ฒ29
ธ30
น31
ฤ32
ล33
ห34
ฬ35
า36
ำ37
?38
฿39
ๅ40
ๆ41
์42
ํ43
?44
?45
46
47
48
49
ก50
ข51
ฒ
,regularization_losses
ฌnon_trainable_variables
ญlayers
ฎmetrics
-	variables
 ฏlayer_regularization_losses
.trainable_variables
ฐlayer_metrics
 
[Y
VARIABLE_VALUEConv2D_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv2D_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

10
21

10
21
ฒ
4regularization_losses
ฑnon_trainable_variables
ฒlayers
ณmetrics
5	variables
 ดlayer_regularization_losses
6trainable_variables
ตlayer_metrics
 
 
 
 
ฒ
9regularization_losses
ถnon_trainable_variables
ทlayers
ธmetrics
:	variables
 นlayer_regularization_losses
;trainable_variables
บlayer_metrics
_]
VARIABLE_VALUESqueezeFire2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUESqueezeFire2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

=0
>1

=0
>1
ฒ
@regularization_losses
ปnon_trainable_variables
ผlayers
ฝmetrics
A	variables
 พlayer_regularization_losses
Btrainable_variables
ฟlayer_metrics
a_
VARIABLE_VALUEExpand1x1Fire2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEExpand1x1Fire2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

D0
E1

D0
E1
ฒ
Gregularization_losses
ภnon_trainable_variables
มlayers
ยmetrics
H	variables
 รlayer_regularization_losses
Itrainable_variables
ฤlayer_metrics
a_
VARIABLE_VALUEExpand3x3Fire2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEExpand3x3Fire2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

K0
L1

K0
L1
ฒ
Nregularization_losses
ลnon_trainable_variables
ฦlayers
วmetrics
O	variables
 ศlayer_regularization_losses
Ptrainable_variables
ษlayer_metrics
 
 
 
 
ฒ
Sregularization_losses
สnon_trainable_variables
หlayers
ฬmetrics
T	variables
 อlayer_regularization_losses
Utrainable_variables
ฮlayer_metrics
_]
VARIABLE_VALUESqueezeFire3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUESqueezeFire3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

W0
X1

W0
X1
ฒ
Zregularization_losses
ฯnon_trainable_variables
ะlayers
ัmetrics
[	variables
 าlayer_regularization_losses
\trainable_variables
ำlayer_metrics
a_
VARIABLE_VALUEExpand1x1Fire3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEExpand1x1Fire3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

^0
_1

^0
_1
ฒ
aregularization_losses
ิnon_trainable_variables
ีlayers
ึmetrics
b	variables
 ืlayer_regularization_losses
ctrainable_variables
ุlayer_metrics
a_
VARIABLE_VALUEExpand3x3Fire3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEExpand3x3Fire3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

e0
f1

e0
f1
ฒ
hregularization_losses
ูnon_trainable_variables
ฺlayers
?metrics
i	variables
 ?layer_regularization_losses
jtrainable_variables
?layer_metrics
 
 
 
 
ฒ
mregularization_losses
?non_trainable_variables
฿layers
เmetrics
n	variables
 แlayer_regularization_losses
otrainable_variables
โlayer_metrics
_]
VARIABLE_VALUESqueezeFire4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUESqueezeFire4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

q0
r1

q0
r1
ฒ
tregularization_losses
ใnon_trainable_variables
ไlayers
ๅmetrics
u	variables
 ๆlayer_regularization_losses
vtrainable_variables
็layer_metrics
a_
VARIABLE_VALUEExpand1x1Fire4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEExpand1x1Fire4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

x0
y1

x0
y1
ฒ
{regularization_losses
่non_trainable_variables
้layers
๊metrics
|	variables
 ๋layer_regularization_losses
}trainable_variables
์layer_metrics
a_
VARIABLE_VALUEExpand3x3Fire4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEExpand3x3Fire4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
ต
regularization_losses
ํnon_trainable_variables
๎layers
๏metrics
	variables
 ๐layer_regularization_losses
trainable_variables
๑layer_metrics
 
 
 
 
ต
regularization_losses
๒non_trainable_variables
๓layers
๔metrics
	variables
 ๕layer_regularization_losses
trainable_variables
๖layer_metrics
 
 
 
 
ต
regularization_losses
๗non_trainable_variables
๘layers
๙metrics
	variables
 ๚layer_regularization_losses
trainable_variables
๛layer_metrics
`^
VARIABLE_VALUESqueezeFire5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUESqueezeFire5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
ต
regularization_losses
?non_trainable_variables
?layers
?metrics
	variables
 ?layer_regularization_losses
trainable_variables
layer_metrics
b`
VARIABLE_VALUEExpand1x1Fire5/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand1x1Fire5/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
ต
regularization_losses
non_trainable_variables
layers
metrics
	variables
 layer_regularization_losses
trainable_variables
layer_metrics
b`
VARIABLE_VALUEExpand3x3Fire5/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand3x3Fire5/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
ต
กregularization_losses
non_trainable_variables
layers
metrics
ข	variables
 layer_regularization_losses
ฃtrainable_variables
layer_metrics
 
 
 
 
ต
ฆregularization_losses
non_trainable_variables
layers
metrics
ง	variables
 layer_regularization_losses
จtrainable_variables
layer_metrics
`^
VARIABLE_VALUESqueezeFire6/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUESqueezeFire6/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ช0
ซ1

ช0
ซ1
ต
ญregularization_losses
non_trainable_variables
layers
metrics
ฎ	variables
 layer_regularization_losses
ฏtrainable_variables
layer_metrics
b`
VARIABLE_VALUEExpand1x1Fire6/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand1x1Fire6/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ฑ0
ฒ1

ฑ0
ฒ1
ต
ดregularization_losses
non_trainable_variables
layers
metrics
ต	variables
 layer_regularization_losses
ถtrainable_variables
layer_metrics
b`
VARIABLE_VALUEExpand3x3Fire6/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand3x3Fire6/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ธ0
น1

ธ0
น1
ต
ปregularization_losses
non_trainable_variables
layers
metrics
ผ	variables
 layer_regularization_losses
ฝtrainable_variables
layer_metrics
 
 
 
 
ต
ภregularization_losses
non_trainable_variables
?layers
กmetrics
ม	variables
 ขlayer_regularization_losses
ยtrainable_variables
ฃlayer_metrics
`^
VARIABLE_VALUESqueezeFire7/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUESqueezeFire7/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ฤ0
ล1

ฤ0
ล1
ต
วregularization_losses
คnon_trainable_variables
ฅlayers
ฆmetrics
ศ	variables
 งlayer_regularization_losses
ษtrainable_variables
จlayer_metrics
b`
VARIABLE_VALUEExpand1x1Fire7/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand1x1Fire7/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ห0
ฬ1

ห0
ฬ1
ต
ฮregularization_losses
ฉnon_trainable_variables
ชlayers
ซmetrics
ฯ	variables
 ฌlayer_regularization_losses
ะtrainable_variables
ญlayer_metrics
b`
VARIABLE_VALUEExpand3x3Fire7/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand3x3Fire7/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

า0
ำ1

า0
ำ1
ต
ีregularization_losses
ฎnon_trainable_variables
ฏlayers
ฐmetrics
ึ	variables
 ฑlayer_regularization_losses
ืtrainable_variables
ฒlayer_metrics
 
 
 
 
ต
ฺregularization_losses
ณnon_trainable_variables
ดlayers
ตmetrics
?	variables
 ถlayer_regularization_losses
?trainable_variables
ทlayer_metrics
`^
VARIABLE_VALUESqueezeFire8/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUESqueezeFire8/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
฿1

?0
฿1
ต
แregularization_losses
ธnon_trainable_variables
นlayers
บmetrics
โ	variables
 ปlayer_regularization_losses
ใtrainable_variables
ผlayer_metrics
b`
VARIABLE_VALUEExpand1x1Fire8/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand1x1Fire8/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ๅ0
ๆ1

ๅ0
ๆ1
ต
่regularization_losses
ฝnon_trainable_variables
พlayers
ฟmetrics
้	variables
 ภlayer_regularization_losses
๊trainable_variables
มlayer_metrics
b`
VARIABLE_VALUEExpand3x3Fire8/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand3x3Fire8/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

์0
ํ1

์0
ํ1
ต
๏regularization_losses
ยnon_trainable_variables
รlayers
ฤmetrics
๐	variables
 ลlayer_regularization_losses
๑trainable_variables
ฦlayer_metrics
 
 
 
 
ต
๔regularization_losses
วnon_trainable_variables
ศlayers
ษmetrics
๕	variables
 สlayer_regularization_losses
๖trainable_variables
หlayer_metrics
 
 
 
 
ต
๙regularization_losses
ฬnon_trainable_variables
อlayers
ฮmetrics
๚	variables
 ฯlayer_regularization_losses
๛trainable_variables
ะlayer_metrics
`^
VARIABLE_VALUESqueezeFire9/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUESqueezeFire9/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1

?0
?1
ต
regularization_losses
ัnon_trainable_variables
าlayers
ำmetrics
	variables
 ิlayer_regularization_losses
trainable_variables
ีlayer_metrics
b`
VARIABLE_VALUEExpand1x1Fire9/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand1x1Fire9/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
ต
regularization_losses
ึnon_trainable_variables
ืlayers
ุmetrics
	variables
 ูlayer_regularization_losses
trainable_variables
ฺlayer_metrics
b`
VARIABLE_VALUEExpand3x3Fire9/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEExpand3x3Fire9/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
ต
regularization_losses
?non_trainable_variables
?layers
?metrics
	variables
 ?layer_regularization_losses
trainable_variables
฿layer_metrics
 
 
 
 
ต
regularization_losses
เnon_trainable_variables
แlayers
โmetrics
	variables
 ใlayer_regularization_losses
trainable_variables
ไlayer_metrics
 
 
 
 
ต
regularization_losses
ๅnon_trainable_variables
ๆlayers
็metrics
	variables
 ่layer_regularization_losses
trainable_variables
้layer_metrics
 
 
 
 
ต
regularization_losses
๊non_trainable_variables
๋layers
์metrics
	variables
 ํlayer_regularization_losses
trainable_variables
๎layer_metrics
^\
VARIABLE_VALUEDenseFinal/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEDenseFinal/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

ก0
ข1

ก0
ข1
ต
คregularization_losses
๏non_trainable_variables
๐layers
๑metrics
ฅ	variables
 ๒layer_regularization_losses
ฆtrainable_variables
๓layer_metrics
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
ถ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39

๔0
๕1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

๖total

๗count
๘	variables
๙	keras_api
I

๚total

๛count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

๖0
๗1

๘	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

๚0
๛1

?	variables

serving_default_InputPlaceholder*1
_output_shapes
:?????????เเ*
dtype0*&
shape:?????????เเ
฿
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputConv2D_1/kernelConv2D_1/biasSqueezeFire2/kernelSqueezeFire2/biasExpand1x1Fire2/kernelExpand1x1Fire2/biasExpand3x3Fire2/kernelExpand3x3Fire2/biasSqueezeFire3/kernelSqueezeFire3/biasExpand1x1Fire3/kernelExpand1x1Fire3/biasExpand3x3Fire3/kernelExpand3x3Fire3/biasSqueezeFire4/kernelSqueezeFire4/biasExpand1x1Fire4/kernelExpand1x1Fire4/biasExpand3x3Fire4/kernelExpand3x3Fire4/biasSqueezeFire5/kernelSqueezeFire5/biasExpand1x1Fire5/kernelExpand1x1Fire5/biasExpand3x3Fire5/kernelExpand3x3Fire5/biasSqueezeFire6/kernelSqueezeFire6/biasExpand1x1Fire6/kernelExpand1x1Fire6/biasExpand3x3Fire6/kernelExpand3x3Fire6/biasSqueezeFire7/kernelSqueezeFire7/biasExpand1x1Fire7/kernelExpand1x1Fire7/biasExpand3x3Fire7/kernelExpand3x3Fire7/biasSqueezeFire8/kernelSqueezeFire8/biasExpand1x1Fire8/kernelExpand1x1Fire8/biasExpand3x3Fire8/kernelExpand3x3Fire8/biasSqueezeFire9/kernelSqueezeFire9/biasExpand1x1Fire9/kernelExpand1x1Fire9/biasExpand3x3Fire9/kernelExpand3x3Fire9/biasDenseFinal/kernelDenseFinal/bias*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_267537
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ค
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv2D_1/kernel/Read/ReadVariableOp!Conv2D_1/bias/Read/ReadVariableOp'SqueezeFire2/kernel/Read/ReadVariableOp%SqueezeFire2/bias/Read/ReadVariableOp)Expand1x1Fire2/kernel/Read/ReadVariableOp'Expand1x1Fire2/bias/Read/ReadVariableOp)Expand3x3Fire2/kernel/Read/ReadVariableOp'Expand3x3Fire2/bias/Read/ReadVariableOp'SqueezeFire3/kernel/Read/ReadVariableOp%SqueezeFire3/bias/Read/ReadVariableOp)Expand1x1Fire3/kernel/Read/ReadVariableOp'Expand1x1Fire3/bias/Read/ReadVariableOp)Expand3x3Fire3/kernel/Read/ReadVariableOp'Expand3x3Fire3/bias/Read/ReadVariableOp'SqueezeFire4/kernel/Read/ReadVariableOp%SqueezeFire4/bias/Read/ReadVariableOp)Expand1x1Fire4/kernel/Read/ReadVariableOp'Expand1x1Fire4/bias/Read/ReadVariableOp)Expand3x3Fire4/kernel/Read/ReadVariableOp'Expand3x3Fire4/bias/Read/ReadVariableOp'SqueezeFire5/kernel/Read/ReadVariableOp%SqueezeFire5/bias/Read/ReadVariableOp)Expand1x1Fire5/kernel/Read/ReadVariableOp'Expand1x1Fire5/bias/Read/ReadVariableOp)Expand3x3Fire5/kernel/Read/ReadVariableOp'Expand3x3Fire5/bias/Read/ReadVariableOp'SqueezeFire6/kernel/Read/ReadVariableOp%SqueezeFire6/bias/Read/ReadVariableOp)Expand1x1Fire6/kernel/Read/ReadVariableOp'Expand1x1Fire6/bias/Read/ReadVariableOp)Expand3x3Fire6/kernel/Read/ReadVariableOp'Expand3x3Fire6/bias/Read/ReadVariableOp'SqueezeFire7/kernel/Read/ReadVariableOp%SqueezeFire7/bias/Read/ReadVariableOp)Expand1x1Fire7/kernel/Read/ReadVariableOp'Expand1x1Fire7/bias/Read/ReadVariableOp)Expand3x3Fire7/kernel/Read/ReadVariableOp'Expand3x3Fire7/bias/Read/ReadVariableOp'SqueezeFire8/kernel/Read/ReadVariableOp%SqueezeFire8/bias/Read/ReadVariableOp)Expand1x1Fire8/kernel/Read/ReadVariableOp'Expand1x1Fire8/bias/Read/ReadVariableOp)Expand3x3Fire8/kernel/Read/ReadVariableOp'Expand3x3Fire8/bias/Read/ReadVariableOp'SqueezeFire9/kernel/Read/ReadVariableOp%SqueezeFire9/bias/Read/ReadVariableOp)Expand1x1Fire9/kernel/Read/ReadVariableOp'Expand1x1Fire9/bias/Read/ReadVariableOp)Expand3x3Fire9/kernel/Read/ReadVariableOp'Expand3x3Fire9/bias/Read/ReadVariableOp%DenseFinal/kernel/Read/ReadVariableOp#DenseFinal/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*I
TinB
@2>	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_269918
๏
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv2D_1/kernelConv2D_1/biasSqueezeFire2/kernelSqueezeFire2/biasExpand1x1Fire2/kernelExpand1x1Fire2/biasExpand3x3Fire2/kernelExpand3x3Fire2/biasSqueezeFire3/kernelSqueezeFire3/biasExpand1x1Fire3/kernelExpand1x1Fire3/biasExpand3x3Fire3/kernelExpand3x3Fire3/biasSqueezeFire4/kernelSqueezeFire4/biasExpand1x1Fire4/kernelExpand1x1Fire4/biasExpand3x3Fire4/kernelExpand3x3Fire4/biasSqueezeFire5/kernelSqueezeFire5/biasExpand1x1Fire5/kernelExpand1x1Fire5/biasExpand3x3Fire5/kernelExpand3x3Fire5/biasSqueezeFire6/kernelSqueezeFire6/biasExpand1x1Fire6/kernelExpand1x1Fire6/biasExpand3x3Fire6/kernelExpand3x3Fire6/biasSqueezeFire7/kernelSqueezeFire7/biasExpand1x1Fire7/kernelExpand1x1Fire7/biasExpand3x3Fire7/kernelExpand3x3Fire7/biasSqueezeFire8/kernelSqueezeFire8/biasExpand1x1Fire8/kernelExpand1x1Fire8/biasExpand3x3Fire8/kernelExpand3x3Fire8/biasSqueezeFire9/kernelSqueezeFire9/biasExpand1x1Fire9/kernelExpand1x1Fire9/biasExpand3x3Fire9/kernelExpand3x3Fire9/biasDenseFinal/kernelDenseFinal/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_270108ฟ๋$
ภ	
g
__inference_loss_fn_17_2696385
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ภ	
g
__inference_loss_fn_15_2696165
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ฝ
ฐ
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_265724

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
บ
ฒ
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_268597

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77:::W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs


-__inference_SqueezeFire9_layer_call_fn_269305

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_2658402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_21_2696825
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:


/__inference_Expand1x1Fire9_layer_call_fn_269337

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_2658732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_14_2696055
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ภ	
g
__inference_loss_fn_19_2696605
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ฝ
ฐ
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_265494

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????02
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_20_2696715
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:


/__inference_Expand3x3Fire6_layer_call_fn_269042

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_2655602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
๚
`
D__inference_MaxPool8_layer_call_and_return_conditional_losses_264972

inputs
identityญ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ฟ	
f
__inference_loss_fn_7_2695285
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
น
ฎ
F__inference_DenseFinal_layer_call_and_return_conditional_losses_265993

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ค*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????ค:::Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs


/__inference_Expand3x3Fire8_layer_call_fn_269260

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_2657902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_268860

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ข
E
)__inference_MaxPool4_layer_call_fn_264966

inputs
identity่
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool4_layer_call_and_return_conditional_losses_2649602
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ุ
Y
-__inference_Concatenate8_layer_call_fn_269273
inputs_0
inputs_1
identity฿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate8_layer_call_and_return_conditional_losses_2658132
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ภ	
g
__inference_loss_fn_23_2697045
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:


/__inference_Expand3x3Fire3_layer_call_fn_268715

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_2652142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
ล
b
)__inference_Dropout9_layer_call_fn_269404

inputs
identityขStatefulPartitionedCallๆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dropout9_layer_call_and_return_conditional_losses_2659502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
่

+__inference_DenseFinal_layer_call_fn_269440

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_DenseFinal_layer_call_and_return_conditional_losses_2659932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????ค::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_265560

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
๓
t
H__inference_Concatenate8_layer_call_and_return_conditional_losses_269267
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1


/__inference_Expand3x3Fire7_layer_call_fn_269151

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_2656752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs


/__inference_Expand1x1Fire5_layer_call_fn_268901

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_2654122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_269360

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ุ
Y
-__inference_Concatenate5_layer_call_fn_268946
inputs_0
inputs_1
identity฿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate5_layer_call_and_return_conditional_losses_2654682
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1

?
5__inference_SqueezeNet_Preloaded_layer_call_fn_267270	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identityขStatefulPartitionedCallต
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_2671632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:?????????เเ

_user_specified_nameInput
ธ
ฐ
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_268533

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????772
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77@:::W S
/
_output_shapes
:?????????77@
 
_user_specified_nameinputs
ข
E
)__inference_MaxPool8_layer_call_fn_264978

inputs
identity่
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool8_layer_call_and_return_conditional_losses_2649722
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_268924

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
ฝ	
f
__inference_loss_fn_0_2694515
1kernel_regularizer_square_readvariableop_resource
identityฮ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ย
ฒ
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_265329

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????772
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77 :::W S
/
_output_shapes
:?????????77 
 
_user_specified_nameinputs
๋
b
D__inference_Dropout9_layer_call_and_return_conditional_losses_269399

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_265873

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_24_2697155
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ม
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_269415

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? R 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:?????????ค2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:?????????ค2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_269033

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_265148

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????772
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????77:::X T
0
_output_shapes
:?????????77
 
_user_specified_nameinputs
๎ำ
ล
!__inference__wrapped_model_264942	
input@
<squeezenet_preloaded_conv2d_1_conv2d_readvariableop_resourceA
=squeezenet_preloaded_conv2d_1_biasadd_readvariableop_resourceD
@squeezenet_preloaded_squeezefire2_conv2d_readvariableop_resourceE
Asqueezenet_preloaded_squeezefire2_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand1x1fire2_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand1x1fire2_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand3x3fire2_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand3x3fire2_biasadd_readvariableop_resourceD
@squeezenet_preloaded_squeezefire3_conv2d_readvariableop_resourceE
Asqueezenet_preloaded_squeezefire3_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand1x1fire3_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand1x1fire3_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand3x3fire3_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand3x3fire3_biasadd_readvariableop_resourceD
@squeezenet_preloaded_squeezefire4_conv2d_readvariableop_resourceE
Asqueezenet_preloaded_squeezefire4_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand1x1fire4_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand1x1fire4_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand3x3fire4_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand3x3fire4_biasadd_readvariableop_resourceD
@squeezenet_preloaded_squeezefire5_conv2d_readvariableop_resourceE
Asqueezenet_preloaded_squeezefire5_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand1x1fire5_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand1x1fire5_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand3x3fire5_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand3x3fire5_biasadd_readvariableop_resourceD
@squeezenet_preloaded_squeezefire6_conv2d_readvariableop_resourceE
Asqueezenet_preloaded_squeezefire6_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand1x1fire6_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand1x1fire6_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand3x3fire6_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand3x3fire6_biasadd_readvariableop_resourceD
@squeezenet_preloaded_squeezefire7_conv2d_readvariableop_resourceE
Asqueezenet_preloaded_squeezefire7_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand1x1fire7_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand1x1fire7_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand3x3fire7_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand3x3fire7_biasadd_readvariableop_resourceD
@squeezenet_preloaded_squeezefire8_conv2d_readvariableop_resourceE
Asqueezenet_preloaded_squeezefire8_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand1x1fire8_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand1x1fire8_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand3x3fire8_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand3x3fire8_biasadd_readvariableop_resourceD
@squeezenet_preloaded_squeezefire9_conv2d_readvariableop_resourceE
Asqueezenet_preloaded_squeezefire9_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand1x1fire9_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand1x1fire9_biasadd_readvariableop_resourceF
Bsqueezenet_preloaded_expand3x3fire9_conv2d_readvariableop_resourceG
Csqueezenet_preloaded_expand3x3fire9_biasadd_readvariableop_resourceB
>squeezenet_preloaded_densefinal_matmul_readvariableop_resourceC
?squeezenet_preloaded_densefinal_biasadd_readvariableop_resource
identity๏
3SqueezeNet_Preloaded/Conv2D_1/Conv2D/ReadVariableOpReadVariableOp<squeezenet_preloaded_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype025
3SqueezeNet_Preloaded/Conv2D_1/Conv2D/ReadVariableOp?
$SqueezeNet_Preloaded/Conv2D_1/Conv2DConv2Dinput;SqueezeNet_Preloaded/Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@*
paddingSAME*
strides
2&
$SqueezeNet_Preloaded/Conv2D_1/Conv2Dๆ
4SqueezeNet_Preloaded/Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp=squeezenet_preloaded_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4SqueezeNet_Preloaded/Conv2D_1/BiasAdd/ReadVariableOp
%SqueezeNet_Preloaded/Conv2D_1/BiasAddBiasAdd-SqueezeNet_Preloaded/Conv2D_1/Conv2D:output:0<SqueezeNet_Preloaded/Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@2'
%SqueezeNet_Preloaded/Conv2D_1/BiasAddบ
"SqueezeNet_Preloaded/Conv2D_1/ReluRelu.SqueezeNet_Preloaded/Conv2D_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????pp@2$
"SqueezeNet_Preloaded/Conv2D_1/Relu๘
%SqueezeNet_Preloaded/MaxPool1/MaxPoolMaxPool0SqueezeNet_Preloaded/Conv2D_1/Relu:activations:0*/
_output_shapes
:?????????77@*
ksize
*
paddingVALID*
strides
2'
%SqueezeNet_Preloaded/MaxPool1/MaxPool๛
7SqueezeNet_Preloaded/SqueezeFire2/Conv2D/ReadVariableOpReadVariableOp@squeezenet_preloaded_squeezefire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype029
7SqueezeNet_Preloaded/SqueezeFire2/Conv2D/ReadVariableOpฑ
(SqueezeNet_Preloaded/SqueezeFire2/Conv2DConv2D.SqueezeNet_Preloaded/MaxPool1/MaxPool:output:0?SqueezeNet_Preloaded/SqueezeFire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2*
(SqueezeNet_Preloaded/SqueezeFire2/Conv2D๒
8SqueezeNet_Preloaded/SqueezeFire2/BiasAdd/ReadVariableOpReadVariableOpAsqueezenet_preloaded_squeezefire2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8SqueezeNet_Preloaded/SqueezeFire2/BiasAdd/ReadVariableOp
)SqueezeNet_Preloaded/SqueezeFire2/BiasAddBiasAdd1SqueezeNet_Preloaded/SqueezeFire2/Conv2D:output:0@SqueezeNet_Preloaded/SqueezeFire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772+
)SqueezeNet_Preloaded/SqueezeFire2/BiasAddฦ
&SqueezeNet_Preloaded/SqueezeFire2/ReluRelu2SqueezeNet_Preloaded/SqueezeFire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????772(
&SqueezeNet_Preloaded/SqueezeFire2/Relu
9SqueezeNet_Preloaded/Expand1x1Fire2/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand1x1fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02;
9SqueezeNet_Preloaded/Expand1x1Fire2/Conv2D/ReadVariableOpฝ
*SqueezeNet_Preloaded/Expand1x1Fire2/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire2/Relu:activations:0ASqueezeNet_Preloaded/Expand1x1Fire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand1x1Fire2/Conv2D๘
:SqueezeNet_Preloaded/Expand1x1Fire2/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand1x1fire2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:SqueezeNet_Preloaded/Expand1x1Fire2/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand1x1Fire2/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand1x1Fire2/Conv2D:output:0BSqueezeNet_Preloaded/Expand1x1Fire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2-
+SqueezeNet_Preloaded/Expand1x1Fire2/BiasAddฬ
(SqueezeNet_Preloaded/Expand1x1Fire2/ReluRelu4SqueezeNet_Preloaded/Expand1x1Fire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2*
(SqueezeNet_Preloaded/Expand1x1Fire2/Relu
9SqueezeNet_Preloaded/Expand3x3Fire2/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand3x3fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02;
9SqueezeNet_Preloaded/Expand3x3Fire2/Conv2D/ReadVariableOpฝ
*SqueezeNet_Preloaded/Expand3x3Fire2/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire2/Relu:activations:0ASqueezeNet_Preloaded/Expand3x3Fire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand3x3Fire2/Conv2D๘
:SqueezeNet_Preloaded/Expand3x3Fire2/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand3x3fire2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:SqueezeNet_Preloaded/Expand3x3Fire2/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand3x3Fire2/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand3x3Fire2/Conv2D:output:0BSqueezeNet_Preloaded/Expand3x3Fire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2-
+SqueezeNet_Preloaded/Expand3x3Fire2/BiasAddฬ
(SqueezeNet_Preloaded/Expand3x3Fire2/ReluRelu4SqueezeNet_Preloaded/Expand3x3Fire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2*
(SqueezeNet_Preloaded/Expand3x3Fire2/Relu?
-SqueezeNet_Preloaded/Concatenate2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-SqueezeNet_Preloaded/Concatenate2/concat/axisฬ
(SqueezeNet_Preloaded/Concatenate2/concatConcatV26SqueezeNet_Preloaded/Expand1x1Fire2/Relu:activations:06SqueezeNet_Preloaded/Expand3x3Fire2/Relu:activations:06SqueezeNet_Preloaded/Concatenate2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772*
(SqueezeNet_Preloaded/Concatenate2/concat?
7SqueezeNet_Preloaded/SqueezeFire3/Conv2D/ReadVariableOpReadVariableOp@squeezenet_preloaded_squeezefire3_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype029
7SqueezeNet_Preloaded/SqueezeFire3/Conv2D/ReadVariableOpด
(SqueezeNet_Preloaded/SqueezeFire3/Conv2DConv2D1SqueezeNet_Preloaded/Concatenate2/concat:output:0?SqueezeNet_Preloaded/SqueezeFire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2*
(SqueezeNet_Preloaded/SqueezeFire3/Conv2D๒
8SqueezeNet_Preloaded/SqueezeFire3/BiasAdd/ReadVariableOpReadVariableOpAsqueezenet_preloaded_squeezefire3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8SqueezeNet_Preloaded/SqueezeFire3/BiasAdd/ReadVariableOp
)SqueezeNet_Preloaded/SqueezeFire3/BiasAddBiasAdd1SqueezeNet_Preloaded/SqueezeFire3/Conv2D:output:0@SqueezeNet_Preloaded/SqueezeFire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772+
)SqueezeNet_Preloaded/SqueezeFire3/BiasAddฦ
&SqueezeNet_Preloaded/SqueezeFire3/ReluRelu2SqueezeNet_Preloaded/SqueezeFire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????772(
&SqueezeNet_Preloaded/SqueezeFire3/Relu
9SqueezeNet_Preloaded/Expand1x1Fire3/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand1x1fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02;
9SqueezeNet_Preloaded/Expand1x1Fire3/Conv2D/ReadVariableOpฝ
*SqueezeNet_Preloaded/Expand1x1Fire3/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire3/Relu:activations:0ASqueezeNet_Preloaded/Expand1x1Fire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand1x1Fire3/Conv2D๘
:SqueezeNet_Preloaded/Expand1x1Fire3/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand1x1fire3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:SqueezeNet_Preloaded/Expand1x1Fire3/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand1x1Fire3/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand1x1Fire3/Conv2D:output:0BSqueezeNet_Preloaded/Expand1x1Fire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2-
+SqueezeNet_Preloaded/Expand1x1Fire3/BiasAddฬ
(SqueezeNet_Preloaded/Expand1x1Fire3/ReluRelu4SqueezeNet_Preloaded/Expand1x1Fire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2*
(SqueezeNet_Preloaded/Expand1x1Fire3/Relu
9SqueezeNet_Preloaded/Expand3x3Fire3/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand3x3fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02;
9SqueezeNet_Preloaded/Expand3x3Fire3/Conv2D/ReadVariableOpฝ
*SqueezeNet_Preloaded/Expand3x3Fire3/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire3/Relu:activations:0ASqueezeNet_Preloaded/Expand3x3Fire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand3x3Fire3/Conv2D๘
:SqueezeNet_Preloaded/Expand3x3Fire3/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand3x3fire3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:SqueezeNet_Preloaded/Expand3x3Fire3/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand3x3Fire3/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand3x3Fire3/Conv2D:output:0BSqueezeNet_Preloaded/Expand3x3Fire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2-
+SqueezeNet_Preloaded/Expand3x3Fire3/BiasAddฬ
(SqueezeNet_Preloaded/Expand3x3Fire3/ReluRelu4SqueezeNet_Preloaded/Expand3x3Fire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2*
(SqueezeNet_Preloaded/Expand3x3Fire3/Relu?
-SqueezeNet_Preloaded/Concatenate3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-SqueezeNet_Preloaded/Concatenate3/concat/axisฬ
(SqueezeNet_Preloaded/Concatenate3/concatConcatV26SqueezeNet_Preloaded/Expand1x1Fire3/Relu:activations:06SqueezeNet_Preloaded/Expand3x3Fire3/Relu:activations:06SqueezeNet_Preloaded/Concatenate3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772*
(SqueezeNet_Preloaded/Concatenate3/concat?
7SqueezeNet_Preloaded/SqueezeFire4/Conv2D/ReadVariableOpReadVariableOp@squeezenet_preloaded_squeezefire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype029
7SqueezeNet_Preloaded/SqueezeFire4/Conv2D/ReadVariableOpด
(SqueezeNet_Preloaded/SqueezeFire4/Conv2DConv2D1SqueezeNet_Preloaded/Concatenate3/concat:output:0?SqueezeNet_Preloaded/SqueezeFire4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 *
paddingSAME*
strides
2*
(SqueezeNet_Preloaded/SqueezeFire4/Conv2D๒
8SqueezeNet_Preloaded/SqueezeFire4/BiasAdd/ReadVariableOpReadVariableOpAsqueezenet_preloaded_squeezefire4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8SqueezeNet_Preloaded/SqueezeFire4/BiasAdd/ReadVariableOp
)SqueezeNet_Preloaded/SqueezeFire4/BiasAddBiasAdd1SqueezeNet_Preloaded/SqueezeFire4/Conv2D:output:0@SqueezeNet_Preloaded/SqueezeFire4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 2+
)SqueezeNet_Preloaded/SqueezeFire4/BiasAddฦ
&SqueezeNet_Preloaded/SqueezeFire4/ReluRelu2SqueezeNet_Preloaded/SqueezeFire4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77 2(
&SqueezeNet_Preloaded/SqueezeFire4/Relu
9SqueezeNet_Preloaded/Expand1x1Fire4/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand1x1fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02;
9SqueezeNet_Preloaded/Expand1x1Fire4/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand1x1Fire4/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire4/Relu:activations:0ASqueezeNet_Preloaded/Expand1x1Fire4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand1x1Fire4/Conv2D๙
:SqueezeNet_Preloaded/Expand1x1Fire4/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand1x1fire4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SqueezeNet_Preloaded/Expand1x1Fire4/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand1x1Fire4/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand1x1Fire4/Conv2D:output:0BSqueezeNet_Preloaded/Expand1x1Fire4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772-
+SqueezeNet_Preloaded/Expand1x1Fire4/BiasAddอ
(SqueezeNet_Preloaded/Expand1x1Fire4/ReluRelu4SqueezeNet_Preloaded/Expand1x1Fire4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????772*
(SqueezeNet_Preloaded/Expand1x1Fire4/Relu
9SqueezeNet_Preloaded/Expand3x3Fire4/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand3x3fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02;
9SqueezeNet_Preloaded/Expand3x3Fire4/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand3x3Fire4/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire4/Relu:activations:0ASqueezeNet_Preloaded/Expand3x3Fire4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand3x3Fire4/Conv2D๙
:SqueezeNet_Preloaded/Expand3x3Fire4/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand3x3fire4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SqueezeNet_Preloaded/Expand3x3Fire4/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand3x3Fire4/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand3x3Fire4/Conv2D:output:0BSqueezeNet_Preloaded/Expand3x3Fire4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772-
+SqueezeNet_Preloaded/Expand3x3Fire4/BiasAddอ
(SqueezeNet_Preloaded/Expand3x3Fire4/ReluRelu4SqueezeNet_Preloaded/Expand3x3Fire4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????772*
(SqueezeNet_Preloaded/Expand3x3Fire4/Relu?
-SqueezeNet_Preloaded/Concatenate4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-SqueezeNet_Preloaded/Concatenate4/concat/axisฬ
(SqueezeNet_Preloaded/Concatenate4/concatConcatV26SqueezeNet_Preloaded/Expand1x1Fire4/Relu:activations:06SqueezeNet_Preloaded/Expand3x3Fire4/Relu:activations:06SqueezeNet_Preloaded/Concatenate4/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772*
(SqueezeNet_Preloaded/Concatenate4/concat๚
%SqueezeNet_Preloaded/MaxPool4/MaxPoolMaxPool1SqueezeNet_Preloaded/Concatenate4/concat:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2'
%SqueezeNet_Preloaded/MaxPool4/MaxPool?
7SqueezeNet_Preloaded/SqueezeFire5/Conv2D/ReadVariableOpReadVariableOp@squeezenet_preloaded_squeezefire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype029
7SqueezeNet_Preloaded/SqueezeFire5/Conv2D/ReadVariableOpฑ
(SqueezeNet_Preloaded/SqueezeFire5/Conv2DConv2D.SqueezeNet_Preloaded/MaxPool4/MaxPool:output:0?SqueezeNet_Preloaded/SqueezeFire5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2*
(SqueezeNet_Preloaded/SqueezeFire5/Conv2D๒
8SqueezeNet_Preloaded/SqueezeFire5/BiasAdd/ReadVariableOpReadVariableOpAsqueezenet_preloaded_squeezefire5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8SqueezeNet_Preloaded/SqueezeFire5/BiasAdd/ReadVariableOp
)SqueezeNet_Preloaded/SqueezeFire5/BiasAddBiasAdd1SqueezeNet_Preloaded/SqueezeFire5/Conv2D:output:0@SqueezeNet_Preloaded/SqueezeFire5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2+
)SqueezeNet_Preloaded/SqueezeFire5/BiasAddฦ
&SqueezeNet_Preloaded/SqueezeFire5/ReluRelu2SqueezeNet_Preloaded/SqueezeFire5/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2(
&SqueezeNet_Preloaded/SqueezeFire5/Relu
9SqueezeNet_Preloaded/Expand1x1Fire5/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand1x1fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02;
9SqueezeNet_Preloaded/Expand1x1Fire5/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand1x1Fire5/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire5/Relu:activations:0ASqueezeNet_Preloaded/Expand1x1Fire5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand1x1Fire5/Conv2D๙
:SqueezeNet_Preloaded/Expand1x1Fire5/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand1x1fire5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SqueezeNet_Preloaded/Expand1x1Fire5/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand1x1Fire5/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand1x1Fire5/Conv2D:output:0BSqueezeNet_Preloaded/Expand1x1Fire5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2-
+SqueezeNet_Preloaded/Expand1x1Fire5/BiasAddอ
(SqueezeNet_Preloaded/Expand1x1Fire5/ReluRelu4SqueezeNet_Preloaded/Expand1x1Fire5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Expand1x1Fire5/Relu
9SqueezeNet_Preloaded/Expand3x3Fire5/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand3x3fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02;
9SqueezeNet_Preloaded/Expand3x3Fire5/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand3x3Fire5/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire5/Relu:activations:0ASqueezeNet_Preloaded/Expand3x3Fire5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand3x3Fire5/Conv2D๙
:SqueezeNet_Preloaded/Expand3x3Fire5/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand3x3fire5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SqueezeNet_Preloaded/Expand3x3Fire5/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand3x3Fire5/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand3x3Fire5/Conv2D:output:0BSqueezeNet_Preloaded/Expand3x3Fire5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2-
+SqueezeNet_Preloaded/Expand3x3Fire5/BiasAddอ
(SqueezeNet_Preloaded/Expand3x3Fire5/ReluRelu4SqueezeNet_Preloaded/Expand3x3Fire5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Expand3x3Fire5/Relu?
-SqueezeNet_Preloaded/Concatenate5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-SqueezeNet_Preloaded/Concatenate5/concat/axisฬ
(SqueezeNet_Preloaded/Concatenate5/concatConcatV26SqueezeNet_Preloaded/Expand1x1Fire5/Relu:activations:06SqueezeNet_Preloaded/Expand3x3Fire5/Relu:activations:06SqueezeNet_Preloaded/Concatenate5/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Concatenate5/concat?
7SqueezeNet_Preloaded/SqueezeFire6/Conv2D/ReadVariableOpReadVariableOp@squeezenet_preloaded_squeezefire6_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype029
7SqueezeNet_Preloaded/SqueezeFire6/Conv2D/ReadVariableOpด
(SqueezeNet_Preloaded/SqueezeFire6/Conv2DConv2D1SqueezeNet_Preloaded/Concatenate5/concat:output:0?SqueezeNet_Preloaded/SqueezeFire6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2*
(SqueezeNet_Preloaded/SqueezeFire6/Conv2D๒
8SqueezeNet_Preloaded/SqueezeFire6/BiasAdd/ReadVariableOpReadVariableOpAsqueezenet_preloaded_squeezefire6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02:
8SqueezeNet_Preloaded/SqueezeFire6/BiasAdd/ReadVariableOp
)SqueezeNet_Preloaded/SqueezeFire6/BiasAddBiasAdd1SqueezeNet_Preloaded/SqueezeFire6/Conv2D:output:0@SqueezeNet_Preloaded/SqueezeFire6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02+
)SqueezeNet_Preloaded/SqueezeFire6/BiasAddฦ
&SqueezeNet_Preloaded/SqueezeFire6/ReluRelu2SqueezeNet_Preloaded/SqueezeFire6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02(
&SqueezeNet_Preloaded/SqueezeFire6/Relu
9SqueezeNet_Preloaded/Expand1x1Fire6/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand1x1fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02;
9SqueezeNet_Preloaded/Expand1x1Fire6/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand1x1Fire6/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire6/Relu:activations:0ASqueezeNet_Preloaded/Expand1x1Fire6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand1x1Fire6/Conv2D๙
:SqueezeNet_Preloaded/Expand1x1Fire6/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand1x1fire6_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02<
:SqueezeNet_Preloaded/Expand1x1Fire6/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand1x1Fire6/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand1x1Fire6/Conv2D:output:0BSqueezeNet_Preloaded/Expand1x1Fire6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2-
+SqueezeNet_Preloaded/Expand1x1Fire6/BiasAddอ
(SqueezeNet_Preloaded/Expand1x1Fire6/ReluRelu4SqueezeNet_Preloaded/Expand1x1Fire6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2*
(SqueezeNet_Preloaded/Expand1x1Fire6/Relu
9SqueezeNet_Preloaded/Expand3x3Fire6/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand3x3fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02;
9SqueezeNet_Preloaded/Expand3x3Fire6/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand3x3Fire6/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire6/Relu:activations:0ASqueezeNet_Preloaded/Expand3x3Fire6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand3x3Fire6/Conv2D๙
:SqueezeNet_Preloaded/Expand3x3Fire6/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand3x3fire6_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02<
:SqueezeNet_Preloaded/Expand3x3Fire6/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand3x3Fire6/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand3x3Fire6/Conv2D:output:0BSqueezeNet_Preloaded/Expand3x3Fire6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2-
+SqueezeNet_Preloaded/Expand3x3Fire6/BiasAddอ
(SqueezeNet_Preloaded/Expand3x3Fire6/ReluRelu4SqueezeNet_Preloaded/Expand3x3Fire6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2*
(SqueezeNet_Preloaded/Expand3x3Fire6/Relu?
-SqueezeNet_Preloaded/Concatenate6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-SqueezeNet_Preloaded/Concatenate6/concat/axisฬ
(SqueezeNet_Preloaded/Concatenate6/concatConcatV26SqueezeNet_Preloaded/Expand1x1Fire6/Relu:activations:06SqueezeNet_Preloaded/Expand3x3Fire6/Relu:activations:06SqueezeNet_Preloaded/Concatenate6/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Concatenate6/concat?
7SqueezeNet_Preloaded/SqueezeFire7/Conv2D/ReadVariableOpReadVariableOp@squeezenet_preloaded_squeezefire7_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype029
7SqueezeNet_Preloaded/SqueezeFire7/Conv2D/ReadVariableOpด
(SqueezeNet_Preloaded/SqueezeFire7/Conv2DConv2D1SqueezeNet_Preloaded/Concatenate6/concat:output:0?SqueezeNet_Preloaded/SqueezeFire7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2*
(SqueezeNet_Preloaded/SqueezeFire7/Conv2D๒
8SqueezeNet_Preloaded/SqueezeFire7/BiasAdd/ReadVariableOpReadVariableOpAsqueezenet_preloaded_squeezefire7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02:
8SqueezeNet_Preloaded/SqueezeFire7/BiasAdd/ReadVariableOp
)SqueezeNet_Preloaded/SqueezeFire7/BiasAddBiasAdd1SqueezeNet_Preloaded/SqueezeFire7/Conv2D:output:0@SqueezeNet_Preloaded/SqueezeFire7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02+
)SqueezeNet_Preloaded/SqueezeFire7/BiasAddฦ
&SqueezeNet_Preloaded/SqueezeFire7/ReluRelu2SqueezeNet_Preloaded/SqueezeFire7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02(
&SqueezeNet_Preloaded/SqueezeFire7/Relu
9SqueezeNet_Preloaded/Expand1x1Fire7/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand1x1fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02;
9SqueezeNet_Preloaded/Expand1x1Fire7/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand1x1Fire7/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire7/Relu:activations:0ASqueezeNet_Preloaded/Expand1x1Fire7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand1x1Fire7/Conv2D๙
:SqueezeNet_Preloaded/Expand1x1Fire7/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand1x1fire7_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02<
:SqueezeNet_Preloaded/Expand1x1Fire7/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand1x1Fire7/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand1x1Fire7/Conv2D:output:0BSqueezeNet_Preloaded/Expand1x1Fire7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2-
+SqueezeNet_Preloaded/Expand1x1Fire7/BiasAddอ
(SqueezeNet_Preloaded/Expand1x1Fire7/ReluRelu4SqueezeNet_Preloaded/Expand1x1Fire7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2*
(SqueezeNet_Preloaded/Expand1x1Fire7/Relu
9SqueezeNet_Preloaded/Expand3x3Fire7/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand3x3fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02;
9SqueezeNet_Preloaded/Expand3x3Fire7/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand3x3Fire7/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire7/Relu:activations:0ASqueezeNet_Preloaded/Expand3x3Fire7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand3x3Fire7/Conv2D๙
:SqueezeNet_Preloaded/Expand3x3Fire7/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand3x3fire7_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02<
:SqueezeNet_Preloaded/Expand3x3Fire7/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand3x3Fire7/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand3x3Fire7/Conv2D:output:0BSqueezeNet_Preloaded/Expand3x3Fire7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2-
+SqueezeNet_Preloaded/Expand3x3Fire7/BiasAddอ
(SqueezeNet_Preloaded/Expand3x3Fire7/ReluRelu4SqueezeNet_Preloaded/Expand3x3Fire7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2*
(SqueezeNet_Preloaded/Expand3x3Fire7/Relu?
-SqueezeNet_Preloaded/Concatenate7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-SqueezeNet_Preloaded/Concatenate7/concat/axisฬ
(SqueezeNet_Preloaded/Concatenate7/concatConcatV26SqueezeNet_Preloaded/Expand1x1Fire7/Relu:activations:06SqueezeNet_Preloaded/Expand3x3Fire7/Relu:activations:06SqueezeNet_Preloaded/Concatenate7/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Concatenate7/concat?
7SqueezeNet_Preloaded/SqueezeFire8/Conv2D/ReadVariableOpReadVariableOp@squeezenet_preloaded_squeezefire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype029
7SqueezeNet_Preloaded/SqueezeFire8/Conv2D/ReadVariableOpด
(SqueezeNet_Preloaded/SqueezeFire8/Conv2DConv2D1SqueezeNet_Preloaded/Concatenate7/concat:output:0?SqueezeNet_Preloaded/SqueezeFire8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2*
(SqueezeNet_Preloaded/SqueezeFire8/Conv2D๒
8SqueezeNet_Preloaded/SqueezeFire8/BiasAdd/ReadVariableOpReadVariableOpAsqueezenet_preloaded_squeezefire8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8SqueezeNet_Preloaded/SqueezeFire8/BiasAdd/ReadVariableOp
)SqueezeNet_Preloaded/SqueezeFire8/BiasAddBiasAdd1SqueezeNet_Preloaded/SqueezeFire8/Conv2D:output:0@SqueezeNet_Preloaded/SqueezeFire8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2+
)SqueezeNet_Preloaded/SqueezeFire8/BiasAddฦ
&SqueezeNet_Preloaded/SqueezeFire8/ReluRelu2SqueezeNet_Preloaded/SqueezeFire8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2(
&SqueezeNet_Preloaded/SqueezeFire8/Relu
9SqueezeNet_Preloaded/Expand1x1Fire8/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand1x1fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02;
9SqueezeNet_Preloaded/Expand1x1Fire8/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand1x1Fire8/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire8/Relu:activations:0ASqueezeNet_Preloaded/Expand1x1Fire8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand1x1Fire8/Conv2D๙
:SqueezeNet_Preloaded/Expand1x1Fire8/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand1x1fire8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SqueezeNet_Preloaded/Expand1x1Fire8/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand1x1Fire8/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand1x1Fire8/Conv2D:output:0BSqueezeNet_Preloaded/Expand1x1Fire8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2-
+SqueezeNet_Preloaded/Expand1x1Fire8/BiasAddอ
(SqueezeNet_Preloaded/Expand1x1Fire8/ReluRelu4SqueezeNet_Preloaded/Expand1x1Fire8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Expand1x1Fire8/Relu
9SqueezeNet_Preloaded/Expand3x3Fire8/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand3x3fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02;
9SqueezeNet_Preloaded/Expand3x3Fire8/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand3x3Fire8/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire8/Relu:activations:0ASqueezeNet_Preloaded/Expand3x3Fire8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand3x3Fire8/Conv2D๙
:SqueezeNet_Preloaded/Expand3x3Fire8/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand3x3fire8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SqueezeNet_Preloaded/Expand3x3Fire8/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand3x3Fire8/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand3x3Fire8/Conv2D:output:0BSqueezeNet_Preloaded/Expand3x3Fire8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2-
+SqueezeNet_Preloaded/Expand3x3Fire8/BiasAddอ
(SqueezeNet_Preloaded/Expand3x3Fire8/ReluRelu4SqueezeNet_Preloaded/Expand3x3Fire8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Expand3x3Fire8/Relu?
-SqueezeNet_Preloaded/Concatenate8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-SqueezeNet_Preloaded/Concatenate8/concat/axisฬ
(SqueezeNet_Preloaded/Concatenate8/concatConcatV26SqueezeNet_Preloaded/Expand1x1Fire8/Relu:activations:06SqueezeNet_Preloaded/Expand3x3Fire8/Relu:activations:06SqueezeNet_Preloaded/Concatenate8/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Concatenate8/concat๚
%SqueezeNet_Preloaded/MaxPool8/MaxPoolMaxPool1SqueezeNet_Preloaded/Concatenate8/concat:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2'
%SqueezeNet_Preloaded/MaxPool8/MaxPool?
7SqueezeNet_Preloaded/SqueezeFire9/Conv2D/ReadVariableOpReadVariableOp@squeezenet_preloaded_squeezefire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype029
7SqueezeNet_Preloaded/SqueezeFire9/Conv2D/ReadVariableOpฑ
(SqueezeNet_Preloaded/SqueezeFire9/Conv2DConv2D.SqueezeNet_Preloaded/MaxPool8/MaxPool:output:0?SqueezeNet_Preloaded/SqueezeFire9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2*
(SqueezeNet_Preloaded/SqueezeFire9/Conv2D๒
8SqueezeNet_Preloaded/SqueezeFire9/BiasAdd/ReadVariableOpReadVariableOpAsqueezenet_preloaded_squeezefire9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8SqueezeNet_Preloaded/SqueezeFire9/BiasAdd/ReadVariableOp
)SqueezeNet_Preloaded/SqueezeFire9/BiasAddBiasAdd1SqueezeNet_Preloaded/SqueezeFire9/Conv2D:output:0@SqueezeNet_Preloaded/SqueezeFire9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2+
)SqueezeNet_Preloaded/SqueezeFire9/BiasAddฦ
&SqueezeNet_Preloaded/SqueezeFire9/ReluRelu2SqueezeNet_Preloaded/SqueezeFire9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2(
&SqueezeNet_Preloaded/SqueezeFire9/Relu
9SqueezeNet_Preloaded/Expand1x1Fire9/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand1x1fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02;
9SqueezeNet_Preloaded/Expand1x1Fire9/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand1x1Fire9/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire9/Relu:activations:0ASqueezeNet_Preloaded/Expand1x1Fire9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand1x1Fire9/Conv2D๙
:SqueezeNet_Preloaded/Expand1x1Fire9/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand1x1fire9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SqueezeNet_Preloaded/Expand1x1Fire9/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand1x1Fire9/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand1x1Fire9/Conv2D:output:0BSqueezeNet_Preloaded/Expand1x1Fire9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2-
+SqueezeNet_Preloaded/Expand1x1Fire9/BiasAddอ
(SqueezeNet_Preloaded/Expand1x1Fire9/ReluRelu4SqueezeNet_Preloaded/Expand1x1Fire9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Expand1x1Fire9/Relu
9SqueezeNet_Preloaded/Expand3x3Fire9/Conv2D/ReadVariableOpReadVariableOpBsqueezenet_preloaded_expand3x3fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02;
9SqueezeNet_Preloaded/Expand3x3Fire9/Conv2D/ReadVariableOpพ
*SqueezeNet_Preloaded/Expand3x3Fire9/Conv2DConv2D4SqueezeNet_Preloaded/SqueezeFire9/Relu:activations:0ASqueezeNet_Preloaded/Expand3x3Fire9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2,
*SqueezeNet_Preloaded/Expand3x3Fire9/Conv2D๙
:SqueezeNet_Preloaded/Expand3x3Fire9/BiasAdd/ReadVariableOpReadVariableOpCsqueezenet_preloaded_expand3x3fire9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SqueezeNet_Preloaded/Expand3x3Fire9/BiasAdd/ReadVariableOp
+SqueezeNet_Preloaded/Expand3x3Fire9/BiasAddBiasAdd3SqueezeNet_Preloaded/Expand3x3Fire9/Conv2D:output:0BSqueezeNet_Preloaded/Expand3x3Fire9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2-
+SqueezeNet_Preloaded/Expand3x3Fire9/BiasAddอ
(SqueezeNet_Preloaded/Expand3x3Fire9/ReluRelu4SqueezeNet_Preloaded/Expand3x3Fire9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Expand3x3Fire9/Relu?
-SqueezeNet_Preloaded/Concatenate9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-SqueezeNet_Preloaded/Concatenate9/concat/axisฬ
(SqueezeNet_Preloaded/Concatenate9/concatConcatV26SqueezeNet_Preloaded/Expand1x1Fire9/Relu:activations:06SqueezeNet_Preloaded/Expand3x3Fire9/Relu:activations:06SqueezeNet_Preloaded/Concatenate9/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2*
(SqueezeNet_Preloaded/Concatenate9/concatส
&SqueezeNet_Preloaded/Dropout9/IdentityIdentity1SqueezeNet_Preloaded/Concatenate9/concat:output:0*
T0*0
_output_shapes
:?????????2(
&SqueezeNet_Preloaded/Dropout9/Identity
$SqueezeNet_Preloaded/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? R 2&
$SqueezeNet_Preloaded/flatten_1/Const๏
&SqueezeNet_Preloaded/flatten_1/ReshapeReshape/SqueezeNet_Preloaded/Dropout9/Identity:output:0-SqueezeNet_Preloaded/flatten_1/Const:output:0*
T0*)
_output_shapes
:?????????ค2(
&SqueezeNet_Preloaded/flatten_1/Reshape๏
5SqueezeNet_Preloaded/DenseFinal/MatMul/ReadVariableOpReadVariableOp>squeezenet_preloaded_densefinal_matmul_readvariableop_resource* 
_output_shapes
:
ค*
dtype027
5SqueezeNet_Preloaded/DenseFinal/MatMul/ReadVariableOp?
&SqueezeNet_Preloaded/DenseFinal/MatMulMatMul/SqueezeNet_Preloaded/flatten_1/Reshape:output:0=SqueezeNet_Preloaded/DenseFinal/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&SqueezeNet_Preloaded/DenseFinal/MatMul์
6SqueezeNet_Preloaded/DenseFinal/BiasAdd/ReadVariableOpReadVariableOp?squeezenet_preloaded_densefinal_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6SqueezeNet_Preloaded/DenseFinal/BiasAdd/ReadVariableOp
'SqueezeNet_Preloaded/DenseFinal/BiasAddBiasAdd0SqueezeNet_Preloaded/DenseFinal/MatMul:product:0>SqueezeNet_Preloaded/DenseFinal/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'SqueezeNet_Preloaded/DenseFinal/BiasAddม
'SqueezeNet_Preloaded/DenseFinal/SoftmaxSoftmax0SqueezeNet_Preloaded/DenseFinal/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2)
'SqueezeNet_Preloaded/DenseFinal/Softmax
IdentityIdentity1SqueezeNet_Preloaded/DenseFinal/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ:::::::::::::::::::::::::::::::::::::::::::::::::::::X T
1
_output_shapes
:?????????เเ

_user_specified_nameInput
๏
t
H__inference_Concatenate2_layer_call_and_return_conditional_losses_268613
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????77@:?????????77@:Y U
/
_output_shapes
:?????????77@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????77@
"
_user_specified_name
inputs/1
ส
c
D__inference_Dropout9_layer_call_and_return_conditional_losses_265950

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeฝ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yว
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


/__inference_Expand3x3Fire5_layer_call_fn_268933

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_2654452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
น
E
)__inference_Dropout9_layer_call_fn_269409

inputs
identityฮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dropout9_layer_call_and_return_conditional_losses_2659552
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
?

$__inference_signature_wrapper_267537	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_2649422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:?????????เเ

_user_specified_nameInput


/__inference_Expand3x3Fire4_layer_call_fn_268824

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_2653292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????77 
 
_user_specified_nameinputs
ฟ	
f
__inference_loss_fn_9_2695505
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:


-__inference_SqueezeFire3_layer_call_fn_268651

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_2651482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????77::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????77
 
_user_specified_nameinputs


/__inference_Expand1x1Fire4_layer_call_fn_268792

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_2652962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????77 
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_269296

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


-__inference_SqueezeFire8_layer_call_fn_269196

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_2657242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_269328

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
๓
t
H__inference_Concatenate9_layer_call_and_return_conditional_losses_269376
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ย
ฒ
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_265527

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
ุ
Y
-__inference_Concatenate6_layer_call_fn_269055
inputs_0
inputs_1
identity฿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate6_layer_call_and_return_conditional_losses_2655832
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????ภ:?????????ภ:Z V
0
_output_shapes
:?????????ภ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????ภ
"
_user_specified_name
inputs/1
ย
ฒ
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_265296

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????772
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77 :::W S
/
_output_shapes
:?????????77 
 
_user_specified_nameinputs
ุ๚
ฦ
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_266457	
input
conv2d_1_266163
conv2d_1_266165
squeezefire2_266169
squeezefire2_266171
expand1x1fire2_266174
expand1x1fire2_266176
expand3x3fire2_266179
expand3x3fire2_266181
squeezefire3_266185
squeezefire3_266187
expand1x1fire3_266190
expand1x1fire3_266192
expand3x3fire3_266195
expand3x3fire3_266197
squeezefire4_266201
squeezefire4_266203
expand1x1fire4_266206
expand1x1fire4_266208
expand3x3fire4_266211
expand3x3fire4_266213
squeezefire5_266218
squeezefire5_266220
expand1x1fire5_266223
expand1x1fire5_266225
expand3x3fire5_266228
expand3x3fire5_266230
squeezefire6_266234
squeezefire6_266236
expand1x1fire6_266239
expand1x1fire6_266241
expand3x3fire6_266244
expand3x3fire6_266246
squeezefire7_266250
squeezefire7_266252
expand1x1fire7_266255
expand1x1fire7_266257
expand3x3fire7_266260
expand3x3fire7_266262
squeezefire8_266266
squeezefire8_266268
expand1x1fire8_266271
expand1x1fire8_266273
expand3x3fire8_266276
expand3x3fire8_266278
squeezefire9_266283
squeezefire9_266285
expand1x1fire9_266288
expand1x1fire9_266290
expand3x3fire9_266293
expand3x3fire9_266295
densefinal_266301
densefinal_266303
identityข Conv2D_1/StatefulPartitionedCallข"DenseFinal/StatefulPartitionedCallข&Expand1x1Fire2/StatefulPartitionedCallข&Expand1x1Fire3/StatefulPartitionedCallข&Expand1x1Fire4/StatefulPartitionedCallข&Expand1x1Fire5/StatefulPartitionedCallข&Expand1x1Fire6/StatefulPartitionedCallข&Expand1x1Fire7/StatefulPartitionedCallข&Expand1x1Fire8/StatefulPartitionedCallข&Expand1x1Fire9/StatefulPartitionedCallข&Expand3x3Fire2/StatefulPartitionedCallข&Expand3x3Fire3/StatefulPartitionedCallข&Expand3x3Fire4/StatefulPartitionedCallข&Expand3x3Fire5/StatefulPartitionedCallข&Expand3x3Fire6/StatefulPartitionedCallข&Expand3x3Fire7/StatefulPartitionedCallข&Expand3x3Fire8/StatefulPartitionedCallข&Expand3x3Fire9/StatefulPartitionedCallข$SqueezeFire2/StatefulPartitionedCallข$SqueezeFire3/StatefulPartitionedCallข$SqueezeFire4/StatefulPartitionedCallข$SqueezeFire5/StatefulPartitionedCallข$SqueezeFire6/StatefulPartitionedCallข$SqueezeFire7/StatefulPartitionedCallข$SqueezeFire8/StatefulPartitionedCallข$SqueezeFire9/StatefulPartitionedCall
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_1_266163conv2d_1_266165*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_2649992"
 Conv2D_1/StatefulPartitionedCall
MaxPool1/PartitionedCallPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool1_layer_call_and_return_conditional_losses_2649482
MaxPool1/PartitionedCallฮ
$SqueezeFire2/StatefulPartitionedCallStatefulPartitionedCall!MaxPool1/PartitionedCall:output:0squeezefire2_266169squeezefire2_266171*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_2650332&
$SqueezeFire2/StatefulPartitionedCallไ
&Expand1x1Fire2/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire2/StatefulPartitionedCall:output:0expand1x1fire2_266174expand1x1fire2_266176*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_2650662(
&Expand1x1Fire2/StatefulPartitionedCallไ
&Expand3x3Fire2/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire2/StatefulPartitionedCall:output:0expand3x3fire2_266179expand3x3fire2_266181*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_2650992(
&Expand3x3Fire2/StatefulPartitionedCallว
Concatenate2/PartitionedCallPartitionedCall/Expand1x1Fire2/StatefulPartitionedCall:output:0/Expand3x3Fire2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate2_layer_call_and_return_conditional_losses_2651222
Concatenate2/PartitionedCallา
$SqueezeFire3/StatefulPartitionedCallStatefulPartitionedCall%Concatenate2/PartitionedCall:output:0squeezefire3_266185squeezefire3_266187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_2651482&
$SqueezeFire3/StatefulPartitionedCallไ
&Expand1x1Fire3/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire3/StatefulPartitionedCall:output:0expand1x1fire3_266190expand1x1fire3_266192*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_2651812(
&Expand1x1Fire3/StatefulPartitionedCallไ
&Expand3x3Fire3/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire3/StatefulPartitionedCall:output:0expand3x3fire3_266195expand3x3fire3_266197*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_2652142(
&Expand3x3Fire3/StatefulPartitionedCallว
Concatenate3/PartitionedCallPartitionedCall/Expand1x1Fire3/StatefulPartitionedCall:output:0/Expand3x3Fire3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate3_layer_call_and_return_conditional_losses_2652372
Concatenate3/PartitionedCallา
$SqueezeFire4/StatefulPartitionedCallStatefulPartitionedCall%Concatenate3/PartitionedCall:output:0squeezefire4_266201squeezefire4_266203*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_2652632&
$SqueezeFire4/StatefulPartitionedCallๅ
&Expand1x1Fire4/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire4/StatefulPartitionedCall:output:0expand1x1fire4_266206expand1x1fire4_266208*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_2652962(
&Expand1x1Fire4/StatefulPartitionedCallๅ
&Expand3x3Fire4/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire4/StatefulPartitionedCall:output:0expand3x3fire4_266211expand3x3fire4_266213*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_2653292(
&Expand3x3Fire4/StatefulPartitionedCallว
Concatenate4/PartitionedCallPartitionedCall/Expand1x1Fire4/StatefulPartitionedCall:output:0/Expand3x3Fire4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate4_layer_call_and_return_conditional_losses_2653522
Concatenate4/PartitionedCall?
MaxPool4/PartitionedCallPartitionedCall%Concatenate4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool4_layer_call_and_return_conditional_losses_2649602
MaxPool4/PartitionedCallฮ
$SqueezeFire5/StatefulPartitionedCallStatefulPartitionedCall!MaxPool4/PartitionedCall:output:0squeezefire5_266218squeezefire5_266220*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_2653792&
$SqueezeFire5/StatefulPartitionedCallๅ
&Expand1x1Fire5/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire5/StatefulPartitionedCall:output:0expand1x1fire5_266223expand1x1fire5_266225*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_2654122(
&Expand1x1Fire5/StatefulPartitionedCallๅ
&Expand3x3Fire5/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire5/StatefulPartitionedCall:output:0expand3x3fire5_266228expand3x3fire5_266230*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_2654452(
&Expand3x3Fire5/StatefulPartitionedCallว
Concatenate5/PartitionedCallPartitionedCall/Expand1x1Fire5/StatefulPartitionedCall:output:0/Expand3x3Fire5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate5_layer_call_and_return_conditional_losses_2654682
Concatenate5/PartitionedCallา
$SqueezeFire6/StatefulPartitionedCallStatefulPartitionedCall%Concatenate5/PartitionedCall:output:0squeezefire6_266234squeezefire6_266236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_2654942&
$SqueezeFire6/StatefulPartitionedCallๅ
&Expand1x1Fire6/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire6/StatefulPartitionedCall:output:0expand1x1fire6_266239expand1x1fire6_266241*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_2655272(
&Expand1x1Fire6/StatefulPartitionedCallๅ
&Expand3x3Fire6/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire6/StatefulPartitionedCall:output:0expand3x3fire6_266244expand3x3fire6_266246*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_2655602(
&Expand3x3Fire6/StatefulPartitionedCallว
Concatenate6/PartitionedCallPartitionedCall/Expand1x1Fire6/StatefulPartitionedCall:output:0/Expand3x3Fire6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate6_layer_call_and_return_conditional_losses_2655832
Concatenate6/PartitionedCallา
$SqueezeFire7/StatefulPartitionedCallStatefulPartitionedCall%Concatenate6/PartitionedCall:output:0squeezefire7_266250squeezefire7_266252*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_2656092&
$SqueezeFire7/StatefulPartitionedCallๅ
&Expand1x1Fire7/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire7/StatefulPartitionedCall:output:0expand1x1fire7_266255expand1x1fire7_266257*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_2656422(
&Expand1x1Fire7/StatefulPartitionedCallๅ
&Expand3x3Fire7/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire7/StatefulPartitionedCall:output:0expand3x3fire7_266260expand3x3fire7_266262*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_2656752(
&Expand3x3Fire7/StatefulPartitionedCallว
Concatenate7/PartitionedCallPartitionedCall/Expand1x1Fire7/StatefulPartitionedCall:output:0/Expand3x3Fire7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate7_layer_call_and_return_conditional_losses_2656982
Concatenate7/PartitionedCallา
$SqueezeFire8/StatefulPartitionedCallStatefulPartitionedCall%Concatenate7/PartitionedCall:output:0squeezefire8_266266squeezefire8_266268*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_2657242&
$SqueezeFire8/StatefulPartitionedCallๅ
&Expand1x1Fire8/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire8/StatefulPartitionedCall:output:0expand1x1fire8_266271expand1x1fire8_266273*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_2657572(
&Expand1x1Fire8/StatefulPartitionedCallๅ
&Expand3x3Fire8/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire8/StatefulPartitionedCall:output:0expand3x3fire8_266276expand3x3fire8_266278*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_2657902(
&Expand3x3Fire8/StatefulPartitionedCallว
Concatenate8/PartitionedCallPartitionedCall/Expand1x1Fire8/StatefulPartitionedCall:output:0/Expand3x3Fire8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate8_layer_call_and_return_conditional_losses_2658132
Concatenate8/PartitionedCall?
MaxPool8/PartitionedCallPartitionedCall%Concatenate8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool8_layer_call_and_return_conditional_losses_2649722
MaxPool8/PartitionedCallฮ
$SqueezeFire9/StatefulPartitionedCallStatefulPartitionedCall!MaxPool8/PartitionedCall:output:0squeezefire9_266283squeezefire9_266285*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_2658402&
$SqueezeFire9/StatefulPartitionedCallๅ
&Expand1x1Fire9/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire9/StatefulPartitionedCall:output:0expand1x1fire9_266288expand1x1fire9_266290*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_2658732(
&Expand1x1Fire9/StatefulPartitionedCallๅ
&Expand3x3Fire9/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire9/StatefulPartitionedCall:output:0expand3x3fire9_266293expand3x3fire9_266295*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_2659062(
&Expand3x3Fire9/StatefulPartitionedCallว
Concatenate9/PartitionedCallPartitionedCall/Expand1x1Fire9/StatefulPartitionedCall:output:0/Expand3x3Fire9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate9_layer_call_and_return_conditional_losses_2659292
Concatenate9/PartitionedCall?
Dropout9/PartitionedCallPartitionedCall%Concatenate9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dropout9_layer_call_and_return_conditional_losses_2659552
Dropout9/PartitionedCall๗
flatten_1/PartitionedCallPartitionedCall!Dropout9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2659742
flatten_1/PartitionedCallฝ
"DenseFinal/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0densefinal_266301densefinal_266303*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_DenseFinal_layer_call_and_return_conditional_losses_2659932$
"DenseFinal/StatefulPartitionedCallฌ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_266163*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulด
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOpsqueezefire2_266169*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpฉ
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_1/Square
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constข
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_1/mul/xค
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulถ
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOpexpand1x1fire2_266174*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpฉ
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_2/Square
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constข
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_2/mul/xค
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulถ
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOpexpand3x3fire2_266179*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_3/Square/ReadVariableOpฉ
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_3/Square
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_3/Constข
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/Sum}
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_3/mul/xค
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/mulต
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOpsqueezefire3_266185*'
_output_shapes
:*
dtype02,
*kernel/Regularizer_4/Square/ReadVariableOpช
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer_4/Square
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_4/Constข
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/Sum}
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_4/mul/xค
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/mulถ
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOpexpand1x1fire3_266190*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_5/Square/ReadVariableOpฉ
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_5/Square
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_5/Constข
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/Sum}
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_5/mul/xค
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/mulถ
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOpexpand3x3fire3_266195*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_6/Square/ReadVariableOpฉ
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_6/Square
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_6/Constข
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/Sum}
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_6/mul/xค
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/mulต
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOpsqueezefire4_266201*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_7/Square/ReadVariableOpช
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_7/Square
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_7/Constข
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/Sum}
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_7/mul/xค
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/mulท
*kernel/Regularizer_8/Square/ReadVariableOpReadVariableOpexpand1x1fire4_266206*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_8/Square/ReadVariableOpช
kernel/Regularizer_8/SquareSquare2kernel/Regularizer_8/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_8/Square
kernel/Regularizer_8/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_8/Constข
kernel/Regularizer_8/SumSumkernel/Regularizer_8/Square:y:0#kernel/Regularizer_8/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/Sum}
kernel/Regularizer_8/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_8/mul/xค
kernel/Regularizer_8/mulMul#kernel/Regularizer_8/mul/x:output:0!kernel/Regularizer_8/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/mulท
*kernel/Regularizer_9/Square/ReadVariableOpReadVariableOpexpand3x3fire4_266211*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_9/Square/ReadVariableOpช
kernel/Regularizer_9/SquareSquare2kernel/Regularizer_9/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_9/Square
kernel/Regularizer_9/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_9/Constข
kernel/Regularizer_9/SumSumkernel/Regularizer_9/Square:y:0#kernel/Regularizer_9/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/Sum}
kernel/Regularizer_9/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_9/mul/xค
kernel/Regularizer_9/mulMul#kernel/Regularizer_9/mul/x:output:0!kernel/Regularizer_9/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/mulท
+kernel/Regularizer_10/Square/ReadVariableOpReadVariableOpsqueezefire5_266218*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_10/Square/ReadVariableOpญ
kernel/Regularizer_10/SquareSquare3kernel/Regularizer_10/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_10/Square
kernel/Regularizer_10/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_10/Constฆ
kernel/Regularizer_10/SumSum kernel/Regularizer_10/Square:y:0$kernel/Regularizer_10/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/Sum
kernel/Regularizer_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_10/mul/xจ
kernel/Regularizer_10/mulMul$kernel/Regularizer_10/mul/x:output:0"kernel/Regularizer_10/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/mulน
+kernel/Regularizer_11/Square/ReadVariableOpReadVariableOpexpand1x1fire5_266223*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_11/Square/ReadVariableOpญ
kernel/Regularizer_11/SquareSquare3kernel/Regularizer_11/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_11/Square
kernel/Regularizer_11/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_11/Constฆ
kernel/Regularizer_11/SumSum kernel/Regularizer_11/Square:y:0$kernel/Regularizer_11/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/Sum
kernel/Regularizer_11/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_11/mul/xจ
kernel/Regularizer_11/mulMul$kernel/Regularizer_11/mul/x:output:0"kernel/Regularizer_11/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/mulน
+kernel/Regularizer_12/Square/ReadVariableOpReadVariableOpexpand3x3fire5_266228*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_12/Square/ReadVariableOpญ
kernel/Regularizer_12/SquareSquare3kernel/Regularizer_12/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_12/Square
kernel/Regularizer_12/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_12/Constฆ
kernel/Regularizer_12/SumSum kernel/Regularizer_12/Square:y:0$kernel/Regularizer_12/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/Sum
kernel/Regularizer_12/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_12/mul/xจ
kernel/Regularizer_12/mulMul$kernel/Regularizer_12/mul/x:output:0"kernel/Regularizer_12/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/mulท
+kernel/Regularizer_13/Square/ReadVariableOpReadVariableOpsqueezefire6_266234*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_13/Square/ReadVariableOpญ
kernel/Regularizer_13/SquareSquare3kernel/Regularizer_13/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_13/Square
kernel/Regularizer_13/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_13/Constฆ
kernel/Regularizer_13/SumSum kernel/Regularizer_13/Square:y:0$kernel/Regularizer_13/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/Sum
kernel/Regularizer_13/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_13/mul/xจ
kernel/Regularizer_13/mulMul$kernel/Regularizer_13/mul/x:output:0"kernel/Regularizer_13/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/mulน
+kernel/Regularizer_14/Square/ReadVariableOpReadVariableOpexpand1x1fire6_266239*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_14/Square/ReadVariableOpญ
kernel/Regularizer_14/SquareSquare3kernel/Regularizer_14/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_14/Square
kernel/Regularizer_14/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_14/Constฆ
kernel/Regularizer_14/SumSum kernel/Regularizer_14/Square:y:0$kernel/Regularizer_14/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/Sum
kernel/Regularizer_14/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_14/mul/xจ
kernel/Regularizer_14/mulMul$kernel/Regularizer_14/mul/x:output:0"kernel/Regularizer_14/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/mulน
+kernel/Regularizer_15/Square/ReadVariableOpReadVariableOpexpand3x3fire6_266244*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_15/Square/ReadVariableOpญ
kernel/Regularizer_15/SquareSquare3kernel/Regularizer_15/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_15/Square
kernel/Regularizer_15/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_15/Constฆ
kernel/Regularizer_15/SumSum kernel/Regularizer_15/Square:y:0$kernel/Regularizer_15/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/Sum
kernel/Regularizer_15/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_15/mul/xจ
kernel/Regularizer_15/mulMul$kernel/Regularizer_15/mul/x:output:0"kernel/Regularizer_15/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/mulท
+kernel/Regularizer_16/Square/ReadVariableOpReadVariableOpsqueezefire7_266250*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_16/Square/ReadVariableOpญ
kernel/Regularizer_16/SquareSquare3kernel/Regularizer_16/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_16/Square
kernel/Regularizer_16/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_16/Constฆ
kernel/Regularizer_16/SumSum kernel/Regularizer_16/Square:y:0$kernel/Regularizer_16/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/Sum
kernel/Regularizer_16/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_16/mul/xจ
kernel/Regularizer_16/mulMul$kernel/Regularizer_16/mul/x:output:0"kernel/Regularizer_16/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/mulน
+kernel/Regularizer_17/Square/ReadVariableOpReadVariableOpexpand1x1fire7_266255*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_17/Square/ReadVariableOpญ
kernel/Regularizer_17/SquareSquare3kernel/Regularizer_17/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_17/Square
kernel/Regularizer_17/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_17/Constฆ
kernel/Regularizer_17/SumSum kernel/Regularizer_17/Square:y:0$kernel/Regularizer_17/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/Sum
kernel/Regularizer_17/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_17/mul/xจ
kernel/Regularizer_17/mulMul$kernel/Regularizer_17/mul/x:output:0"kernel/Regularizer_17/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/mulน
+kernel/Regularizer_18/Square/ReadVariableOpReadVariableOpexpand3x3fire7_266260*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_18/Square/ReadVariableOpญ
kernel/Regularizer_18/SquareSquare3kernel/Regularizer_18/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_18/Square
kernel/Regularizer_18/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_18/Constฆ
kernel/Regularizer_18/SumSum kernel/Regularizer_18/Square:y:0$kernel/Regularizer_18/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/Sum
kernel/Regularizer_18/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_18/mul/xจ
kernel/Regularizer_18/mulMul$kernel/Regularizer_18/mul/x:output:0"kernel/Regularizer_18/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/mulท
+kernel/Regularizer_19/Square/ReadVariableOpReadVariableOpsqueezefire8_266266*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_19/Square/ReadVariableOpญ
kernel/Regularizer_19/SquareSquare3kernel/Regularizer_19/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_19/Square
kernel/Regularizer_19/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_19/Constฆ
kernel/Regularizer_19/SumSum kernel/Regularizer_19/Square:y:0$kernel/Regularizer_19/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/Sum
kernel/Regularizer_19/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_19/mul/xจ
kernel/Regularizer_19/mulMul$kernel/Regularizer_19/mul/x:output:0"kernel/Regularizer_19/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/mulน
+kernel/Regularizer_20/Square/ReadVariableOpReadVariableOpexpand1x1fire8_266271*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_20/Square/ReadVariableOpญ
kernel/Regularizer_20/SquareSquare3kernel/Regularizer_20/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_20/Square
kernel/Regularizer_20/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_20/Constฆ
kernel/Regularizer_20/SumSum kernel/Regularizer_20/Square:y:0$kernel/Regularizer_20/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/Sum
kernel/Regularizer_20/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_20/mul/xจ
kernel/Regularizer_20/mulMul$kernel/Regularizer_20/mul/x:output:0"kernel/Regularizer_20/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/mulน
+kernel/Regularizer_21/Square/ReadVariableOpReadVariableOpexpand3x3fire8_266276*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_21/Square/ReadVariableOpญ
kernel/Regularizer_21/SquareSquare3kernel/Regularizer_21/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_21/Square
kernel/Regularizer_21/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_21/Constฆ
kernel/Regularizer_21/SumSum kernel/Regularizer_21/Square:y:0$kernel/Regularizer_21/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/Sum
kernel/Regularizer_21/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_21/mul/xจ
kernel/Regularizer_21/mulMul$kernel/Regularizer_21/mul/x:output:0"kernel/Regularizer_21/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/mulท
+kernel/Regularizer_22/Square/ReadVariableOpReadVariableOpsqueezefire9_266283*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_22/Square/ReadVariableOpญ
kernel/Regularizer_22/SquareSquare3kernel/Regularizer_22/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_22/Square
kernel/Regularizer_22/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_22/Constฆ
kernel/Regularizer_22/SumSum kernel/Regularizer_22/Square:y:0$kernel/Regularizer_22/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/Sum
kernel/Regularizer_22/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_22/mul/xจ
kernel/Regularizer_22/mulMul$kernel/Regularizer_22/mul/x:output:0"kernel/Regularizer_22/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/mulน
+kernel/Regularizer_23/Square/ReadVariableOpReadVariableOpexpand1x1fire9_266288*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_23/Square/ReadVariableOpญ
kernel/Regularizer_23/SquareSquare3kernel/Regularizer_23/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_23/Square
kernel/Regularizer_23/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_23/Constฆ
kernel/Regularizer_23/SumSum kernel/Regularizer_23/Square:y:0$kernel/Regularizer_23/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/Sum
kernel/Regularizer_23/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_23/mul/xจ
kernel/Regularizer_23/mulMul$kernel/Regularizer_23/mul/x:output:0"kernel/Regularizer_23/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/mulน
+kernel/Regularizer_24/Square/ReadVariableOpReadVariableOpexpand3x3fire9_266293*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_24/Square/ReadVariableOpญ
kernel/Regularizer_24/SquareSquare3kernel/Regularizer_24/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_24/Square
kernel/Regularizer_24/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_24/Constฆ
kernel/Regularizer_24/SumSum kernel/Regularizer_24/Square:y:0$kernel/Regularizer_24/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/Sum
kernel/Regularizer_24/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_24/mul/xจ
kernel/Regularizer_24/mulMul$kernel/Regularizer_24/mul/x:output:0"kernel/Regularizer_24/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/mul	
IdentityIdentity+DenseFinal/StatefulPartitionedCall:output:0!^Conv2D_1/StatefulPartitionedCall#^DenseFinal/StatefulPartitionedCall'^Expand1x1Fire2/StatefulPartitionedCall'^Expand1x1Fire3/StatefulPartitionedCall'^Expand1x1Fire4/StatefulPartitionedCall'^Expand1x1Fire5/StatefulPartitionedCall'^Expand1x1Fire6/StatefulPartitionedCall'^Expand1x1Fire7/StatefulPartitionedCall'^Expand1x1Fire8/StatefulPartitionedCall'^Expand1x1Fire9/StatefulPartitionedCall'^Expand3x3Fire2/StatefulPartitionedCall'^Expand3x3Fire3/StatefulPartitionedCall'^Expand3x3Fire4/StatefulPartitionedCall'^Expand3x3Fire5/StatefulPartitionedCall'^Expand3x3Fire6/StatefulPartitionedCall'^Expand3x3Fire7/StatefulPartitionedCall'^Expand3x3Fire8/StatefulPartitionedCall'^Expand3x3Fire9/StatefulPartitionedCall%^SqueezeFire2/StatefulPartitionedCall%^SqueezeFire3/StatefulPartitionedCall%^SqueezeFire4/StatefulPartitionedCall%^SqueezeFire5/StatefulPartitionedCall%^SqueezeFire6/StatefulPartitionedCall%^SqueezeFire7/StatefulPartitionedCall%^SqueezeFire8/StatefulPartitionedCall%^SqueezeFire9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2H
"DenseFinal/StatefulPartitionedCall"DenseFinal/StatefulPartitionedCall2P
&Expand1x1Fire2/StatefulPartitionedCall&Expand1x1Fire2/StatefulPartitionedCall2P
&Expand1x1Fire3/StatefulPartitionedCall&Expand1x1Fire3/StatefulPartitionedCall2P
&Expand1x1Fire4/StatefulPartitionedCall&Expand1x1Fire4/StatefulPartitionedCall2P
&Expand1x1Fire5/StatefulPartitionedCall&Expand1x1Fire5/StatefulPartitionedCall2P
&Expand1x1Fire6/StatefulPartitionedCall&Expand1x1Fire6/StatefulPartitionedCall2P
&Expand1x1Fire7/StatefulPartitionedCall&Expand1x1Fire7/StatefulPartitionedCall2P
&Expand1x1Fire8/StatefulPartitionedCall&Expand1x1Fire8/StatefulPartitionedCall2P
&Expand1x1Fire9/StatefulPartitionedCall&Expand1x1Fire9/StatefulPartitionedCall2P
&Expand3x3Fire2/StatefulPartitionedCall&Expand3x3Fire2/StatefulPartitionedCall2P
&Expand3x3Fire3/StatefulPartitionedCall&Expand3x3Fire3/StatefulPartitionedCall2P
&Expand3x3Fire4/StatefulPartitionedCall&Expand3x3Fire4/StatefulPartitionedCall2P
&Expand3x3Fire5/StatefulPartitionedCall&Expand3x3Fire5/StatefulPartitionedCall2P
&Expand3x3Fire6/StatefulPartitionedCall&Expand3x3Fire6/StatefulPartitionedCall2P
&Expand3x3Fire7/StatefulPartitionedCall&Expand3x3Fire7/StatefulPartitionedCall2P
&Expand3x3Fire8/StatefulPartitionedCall&Expand3x3Fire8/StatefulPartitionedCall2P
&Expand3x3Fire9/StatefulPartitionedCall&Expand3x3Fire9/StatefulPartitionedCall2L
$SqueezeFire2/StatefulPartitionedCall$SqueezeFire2/StatefulPartitionedCall2L
$SqueezeFire3/StatefulPartitionedCall$SqueezeFire3/StatefulPartitionedCall2L
$SqueezeFire4/StatefulPartitionedCall$SqueezeFire4/StatefulPartitionedCall2L
$SqueezeFire5/StatefulPartitionedCall$SqueezeFire5/StatefulPartitionedCall2L
$SqueezeFire6/StatefulPartitionedCall$SqueezeFire6/StatefulPartitionedCall2L
$SqueezeFire7/StatefulPartitionedCall$SqueezeFire7/StatefulPartitionedCall2L
$SqueezeFire8/StatefulPartitionedCall$SqueezeFire8/StatefulPartitionedCall2L
$SqueezeFire9/StatefulPartitionedCall$SqueezeFire9/StatefulPartitionedCall:X T
1
_output_shapes
:?????????เเ

_user_specified_nameInput
๋
r
H__inference_Concatenate7_layer_call_and_return_conditional_losses_265698

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????ภ:?????????ภ:X T
0
_output_shapes
:?????????ภ
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????ภ
 
_user_specified_nameinputs
บ
ฒ
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_265099

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77:::W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_269001

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_268783

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????772
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77 :::W S
/
_output_shapes
:?????????77 
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_269251

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
็
r
H__inference_Concatenate2_layer_call_and_return_conditional_losses_265122

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????77@:?????????77@:W S
/
_output_shapes
:?????????77@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????77@
 
_user_specified_nameinputs
ธ
ฌ
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_264999

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????pp@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????pp@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????เเ:::Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_268969

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????02
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
๓
t
H__inference_Concatenate4_layer_call_and_return_conditional_losses_268831
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????77:?????????77:Z V
0
_output_shapes
:?????????77
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????77
"
_user_specified_name
inputs/1
๋
b
D__inference_Dropout9_layer_call_and_return_conditional_losses_265955

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


-__inference_SqueezeFire4_layer_call_fn_268760

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_2652632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????77 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????77::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????77
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_265840

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_265263

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77 2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????77:::X T
0
_output_shapes
:?????????77
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_268815

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????772
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77 :::W S
/
_output_shapes
:?????????77 
 
_user_specified_nameinputs
ญ
F
*__inference_flatten_1_layer_call_fn_269420

inputs
identityศ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2659742
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:?????????ค2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ุ
Y
-__inference_Concatenate7_layer_call_fn_269164
inputs_0
inputs_1
identity฿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate7_layer_call_and_return_conditional_losses_2656982
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????ภ:?????????ภ:Z V
0
_output_shapes
:?????????ภ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????ภ
"
_user_specified_name
inputs/1
ภ	
g
__inference_loss_fn_22_2696935
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?
๊
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_266757

inputs
conv2d_1_266463
conv2d_1_266465
squeezefire2_266469
squeezefire2_266471
expand1x1fire2_266474
expand1x1fire2_266476
expand3x3fire2_266479
expand3x3fire2_266481
squeezefire3_266485
squeezefire3_266487
expand1x1fire3_266490
expand1x1fire3_266492
expand3x3fire3_266495
expand3x3fire3_266497
squeezefire4_266501
squeezefire4_266503
expand1x1fire4_266506
expand1x1fire4_266508
expand3x3fire4_266511
expand3x3fire4_266513
squeezefire5_266518
squeezefire5_266520
expand1x1fire5_266523
expand1x1fire5_266525
expand3x3fire5_266528
expand3x3fire5_266530
squeezefire6_266534
squeezefire6_266536
expand1x1fire6_266539
expand1x1fire6_266541
expand3x3fire6_266544
expand3x3fire6_266546
squeezefire7_266550
squeezefire7_266552
expand1x1fire7_266555
expand1x1fire7_266557
expand3x3fire7_266560
expand3x3fire7_266562
squeezefire8_266566
squeezefire8_266568
expand1x1fire8_266571
expand1x1fire8_266573
expand3x3fire8_266576
expand3x3fire8_266578
squeezefire9_266583
squeezefire9_266585
expand1x1fire9_266588
expand1x1fire9_266590
expand3x3fire9_266593
expand3x3fire9_266595
densefinal_266601
densefinal_266603
identityข Conv2D_1/StatefulPartitionedCallข"DenseFinal/StatefulPartitionedCallข Dropout9/StatefulPartitionedCallข&Expand1x1Fire2/StatefulPartitionedCallข&Expand1x1Fire3/StatefulPartitionedCallข&Expand1x1Fire4/StatefulPartitionedCallข&Expand1x1Fire5/StatefulPartitionedCallข&Expand1x1Fire6/StatefulPartitionedCallข&Expand1x1Fire7/StatefulPartitionedCallข&Expand1x1Fire8/StatefulPartitionedCallข&Expand1x1Fire9/StatefulPartitionedCallข&Expand3x3Fire2/StatefulPartitionedCallข&Expand3x3Fire3/StatefulPartitionedCallข&Expand3x3Fire4/StatefulPartitionedCallข&Expand3x3Fire5/StatefulPartitionedCallข&Expand3x3Fire6/StatefulPartitionedCallข&Expand3x3Fire7/StatefulPartitionedCallข&Expand3x3Fire8/StatefulPartitionedCallข&Expand3x3Fire9/StatefulPartitionedCallข$SqueezeFire2/StatefulPartitionedCallข$SqueezeFire3/StatefulPartitionedCallข$SqueezeFire4/StatefulPartitionedCallข$SqueezeFire5/StatefulPartitionedCallข$SqueezeFire6/StatefulPartitionedCallข$SqueezeFire7/StatefulPartitionedCallข$SqueezeFire8/StatefulPartitionedCallข$SqueezeFire9/StatefulPartitionedCall
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_266463conv2d_1_266465*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_2649992"
 Conv2D_1/StatefulPartitionedCall
MaxPool1/PartitionedCallPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool1_layer_call_and_return_conditional_losses_2649482
MaxPool1/PartitionedCallฮ
$SqueezeFire2/StatefulPartitionedCallStatefulPartitionedCall!MaxPool1/PartitionedCall:output:0squeezefire2_266469squeezefire2_266471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_2650332&
$SqueezeFire2/StatefulPartitionedCallไ
&Expand1x1Fire2/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire2/StatefulPartitionedCall:output:0expand1x1fire2_266474expand1x1fire2_266476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_2650662(
&Expand1x1Fire2/StatefulPartitionedCallไ
&Expand3x3Fire2/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire2/StatefulPartitionedCall:output:0expand3x3fire2_266479expand3x3fire2_266481*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_2650992(
&Expand3x3Fire2/StatefulPartitionedCallว
Concatenate2/PartitionedCallPartitionedCall/Expand1x1Fire2/StatefulPartitionedCall:output:0/Expand3x3Fire2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate2_layer_call_and_return_conditional_losses_2651222
Concatenate2/PartitionedCallา
$SqueezeFire3/StatefulPartitionedCallStatefulPartitionedCall%Concatenate2/PartitionedCall:output:0squeezefire3_266485squeezefire3_266487*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_2651482&
$SqueezeFire3/StatefulPartitionedCallไ
&Expand1x1Fire3/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire3/StatefulPartitionedCall:output:0expand1x1fire3_266490expand1x1fire3_266492*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_2651812(
&Expand1x1Fire3/StatefulPartitionedCallไ
&Expand3x3Fire3/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire3/StatefulPartitionedCall:output:0expand3x3fire3_266495expand3x3fire3_266497*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_2652142(
&Expand3x3Fire3/StatefulPartitionedCallว
Concatenate3/PartitionedCallPartitionedCall/Expand1x1Fire3/StatefulPartitionedCall:output:0/Expand3x3Fire3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate3_layer_call_and_return_conditional_losses_2652372
Concatenate3/PartitionedCallา
$SqueezeFire4/StatefulPartitionedCallStatefulPartitionedCall%Concatenate3/PartitionedCall:output:0squeezefire4_266501squeezefire4_266503*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_2652632&
$SqueezeFire4/StatefulPartitionedCallๅ
&Expand1x1Fire4/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire4/StatefulPartitionedCall:output:0expand1x1fire4_266506expand1x1fire4_266508*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_2652962(
&Expand1x1Fire4/StatefulPartitionedCallๅ
&Expand3x3Fire4/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire4/StatefulPartitionedCall:output:0expand3x3fire4_266511expand3x3fire4_266513*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_2653292(
&Expand3x3Fire4/StatefulPartitionedCallว
Concatenate4/PartitionedCallPartitionedCall/Expand1x1Fire4/StatefulPartitionedCall:output:0/Expand3x3Fire4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate4_layer_call_and_return_conditional_losses_2653522
Concatenate4/PartitionedCall?
MaxPool4/PartitionedCallPartitionedCall%Concatenate4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool4_layer_call_and_return_conditional_losses_2649602
MaxPool4/PartitionedCallฮ
$SqueezeFire5/StatefulPartitionedCallStatefulPartitionedCall!MaxPool4/PartitionedCall:output:0squeezefire5_266518squeezefire5_266520*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_2653792&
$SqueezeFire5/StatefulPartitionedCallๅ
&Expand1x1Fire5/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire5/StatefulPartitionedCall:output:0expand1x1fire5_266523expand1x1fire5_266525*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_2654122(
&Expand1x1Fire5/StatefulPartitionedCallๅ
&Expand3x3Fire5/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire5/StatefulPartitionedCall:output:0expand3x3fire5_266528expand3x3fire5_266530*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_2654452(
&Expand3x3Fire5/StatefulPartitionedCallว
Concatenate5/PartitionedCallPartitionedCall/Expand1x1Fire5/StatefulPartitionedCall:output:0/Expand3x3Fire5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate5_layer_call_and_return_conditional_losses_2654682
Concatenate5/PartitionedCallา
$SqueezeFire6/StatefulPartitionedCallStatefulPartitionedCall%Concatenate5/PartitionedCall:output:0squeezefire6_266534squeezefire6_266536*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_2654942&
$SqueezeFire6/StatefulPartitionedCallๅ
&Expand1x1Fire6/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire6/StatefulPartitionedCall:output:0expand1x1fire6_266539expand1x1fire6_266541*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_2655272(
&Expand1x1Fire6/StatefulPartitionedCallๅ
&Expand3x3Fire6/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire6/StatefulPartitionedCall:output:0expand3x3fire6_266544expand3x3fire6_266546*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_2655602(
&Expand3x3Fire6/StatefulPartitionedCallว
Concatenate6/PartitionedCallPartitionedCall/Expand1x1Fire6/StatefulPartitionedCall:output:0/Expand3x3Fire6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate6_layer_call_and_return_conditional_losses_2655832
Concatenate6/PartitionedCallา
$SqueezeFire7/StatefulPartitionedCallStatefulPartitionedCall%Concatenate6/PartitionedCall:output:0squeezefire7_266550squeezefire7_266552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_2656092&
$SqueezeFire7/StatefulPartitionedCallๅ
&Expand1x1Fire7/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire7/StatefulPartitionedCall:output:0expand1x1fire7_266555expand1x1fire7_266557*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_2656422(
&Expand1x1Fire7/StatefulPartitionedCallๅ
&Expand3x3Fire7/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire7/StatefulPartitionedCall:output:0expand3x3fire7_266560expand3x3fire7_266562*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_2656752(
&Expand3x3Fire7/StatefulPartitionedCallว
Concatenate7/PartitionedCallPartitionedCall/Expand1x1Fire7/StatefulPartitionedCall:output:0/Expand3x3Fire7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate7_layer_call_and_return_conditional_losses_2656982
Concatenate7/PartitionedCallา
$SqueezeFire8/StatefulPartitionedCallStatefulPartitionedCall%Concatenate7/PartitionedCall:output:0squeezefire8_266566squeezefire8_266568*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_2657242&
$SqueezeFire8/StatefulPartitionedCallๅ
&Expand1x1Fire8/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire8/StatefulPartitionedCall:output:0expand1x1fire8_266571expand1x1fire8_266573*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_2657572(
&Expand1x1Fire8/StatefulPartitionedCallๅ
&Expand3x3Fire8/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire8/StatefulPartitionedCall:output:0expand3x3fire8_266576expand3x3fire8_266578*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_2657902(
&Expand3x3Fire8/StatefulPartitionedCallว
Concatenate8/PartitionedCallPartitionedCall/Expand1x1Fire8/StatefulPartitionedCall:output:0/Expand3x3Fire8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate8_layer_call_and_return_conditional_losses_2658132
Concatenate8/PartitionedCall?
MaxPool8/PartitionedCallPartitionedCall%Concatenate8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool8_layer_call_and_return_conditional_losses_2649722
MaxPool8/PartitionedCallฮ
$SqueezeFire9/StatefulPartitionedCallStatefulPartitionedCall!MaxPool8/PartitionedCall:output:0squeezefire9_266583squeezefire9_266585*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_2658402&
$SqueezeFire9/StatefulPartitionedCallๅ
&Expand1x1Fire9/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire9/StatefulPartitionedCall:output:0expand1x1fire9_266588expand1x1fire9_266590*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_2658732(
&Expand1x1Fire9/StatefulPartitionedCallๅ
&Expand3x3Fire9/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire9/StatefulPartitionedCall:output:0expand3x3fire9_266593expand3x3fire9_266595*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_2659062(
&Expand3x3Fire9/StatefulPartitionedCallว
Concatenate9/PartitionedCallPartitionedCall/Expand1x1Fire9/StatefulPartitionedCall:output:0/Expand3x3Fire9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate9_layer_call_and_return_conditional_losses_2659292
Concatenate9/PartitionedCall
 Dropout9/StatefulPartitionedCallStatefulPartitionedCall%Concatenate9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dropout9_layer_call_and_return_conditional_losses_2659502"
 Dropout9/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)Dropout9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2659742
flatten_1/PartitionedCallฝ
"DenseFinal/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0densefinal_266601densefinal_266603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_DenseFinal_layer_call_and_return_conditional_losses_2659932$
"DenseFinal/StatefulPartitionedCallฌ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_266463*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulด
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOpsqueezefire2_266469*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpฉ
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_1/Square
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constข
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_1/mul/xค
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulถ
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOpexpand1x1fire2_266474*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpฉ
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_2/Square
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constข
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_2/mul/xค
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulถ
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOpexpand3x3fire2_266479*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_3/Square/ReadVariableOpฉ
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_3/Square
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_3/Constข
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/Sum}
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_3/mul/xค
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/mulต
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOpsqueezefire3_266485*'
_output_shapes
:*
dtype02,
*kernel/Regularizer_4/Square/ReadVariableOpช
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer_4/Square
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_4/Constข
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/Sum}
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_4/mul/xค
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/mulถ
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOpexpand1x1fire3_266490*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_5/Square/ReadVariableOpฉ
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_5/Square
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_5/Constข
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/Sum}
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_5/mul/xค
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/mulถ
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOpexpand3x3fire3_266495*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_6/Square/ReadVariableOpฉ
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_6/Square
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_6/Constข
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/Sum}
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_6/mul/xค
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/mulต
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOpsqueezefire4_266501*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_7/Square/ReadVariableOpช
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_7/Square
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_7/Constข
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/Sum}
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_7/mul/xค
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/mulท
*kernel/Regularizer_8/Square/ReadVariableOpReadVariableOpexpand1x1fire4_266506*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_8/Square/ReadVariableOpช
kernel/Regularizer_8/SquareSquare2kernel/Regularizer_8/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_8/Square
kernel/Regularizer_8/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_8/Constข
kernel/Regularizer_8/SumSumkernel/Regularizer_8/Square:y:0#kernel/Regularizer_8/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/Sum}
kernel/Regularizer_8/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_8/mul/xค
kernel/Regularizer_8/mulMul#kernel/Regularizer_8/mul/x:output:0!kernel/Regularizer_8/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/mulท
*kernel/Regularizer_9/Square/ReadVariableOpReadVariableOpexpand3x3fire4_266511*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_9/Square/ReadVariableOpช
kernel/Regularizer_9/SquareSquare2kernel/Regularizer_9/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_9/Square
kernel/Regularizer_9/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_9/Constข
kernel/Regularizer_9/SumSumkernel/Regularizer_9/Square:y:0#kernel/Regularizer_9/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/Sum}
kernel/Regularizer_9/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_9/mul/xค
kernel/Regularizer_9/mulMul#kernel/Regularizer_9/mul/x:output:0!kernel/Regularizer_9/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/mulท
+kernel/Regularizer_10/Square/ReadVariableOpReadVariableOpsqueezefire5_266518*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_10/Square/ReadVariableOpญ
kernel/Regularizer_10/SquareSquare3kernel/Regularizer_10/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_10/Square
kernel/Regularizer_10/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_10/Constฆ
kernel/Regularizer_10/SumSum kernel/Regularizer_10/Square:y:0$kernel/Regularizer_10/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/Sum
kernel/Regularizer_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_10/mul/xจ
kernel/Regularizer_10/mulMul$kernel/Regularizer_10/mul/x:output:0"kernel/Regularizer_10/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/mulน
+kernel/Regularizer_11/Square/ReadVariableOpReadVariableOpexpand1x1fire5_266523*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_11/Square/ReadVariableOpญ
kernel/Regularizer_11/SquareSquare3kernel/Regularizer_11/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_11/Square
kernel/Regularizer_11/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_11/Constฆ
kernel/Regularizer_11/SumSum kernel/Regularizer_11/Square:y:0$kernel/Regularizer_11/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/Sum
kernel/Regularizer_11/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_11/mul/xจ
kernel/Regularizer_11/mulMul$kernel/Regularizer_11/mul/x:output:0"kernel/Regularizer_11/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/mulน
+kernel/Regularizer_12/Square/ReadVariableOpReadVariableOpexpand3x3fire5_266528*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_12/Square/ReadVariableOpญ
kernel/Regularizer_12/SquareSquare3kernel/Regularizer_12/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_12/Square
kernel/Regularizer_12/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_12/Constฆ
kernel/Regularizer_12/SumSum kernel/Regularizer_12/Square:y:0$kernel/Regularizer_12/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/Sum
kernel/Regularizer_12/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_12/mul/xจ
kernel/Regularizer_12/mulMul$kernel/Regularizer_12/mul/x:output:0"kernel/Regularizer_12/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/mulท
+kernel/Regularizer_13/Square/ReadVariableOpReadVariableOpsqueezefire6_266534*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_13/Square/ReadVariableOpญ
kernel/Regularizer_13/SquareSquare3kernel/Regularizer_13/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_13/Square
kernel/Regularizer_13/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_13/Constฆ
kernel/Regularizer_13/SumSum kernel/Regularizer_13/Square:y:0$kernel/Regularizer_13/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/Sum
kernel/Regularizer_13/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_13/mul/xจ
kernel/Regularizer_13/mulMul$kernel/Regularizer_13/mul/x:output:0"kernel/Regularizer_13/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/mulน
+kernel/Regularizer_14/Square/ReadVariableOpReadVariableOpexpand1x1fire6_266539*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_14/Square/ReadVariableOpญ
kernel/Regularizer_14/SquareSquare3kernel/Regularizer_14/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_14/Square
kernel/Regularizer_14/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_14/Constฆ
kernel/Regularizer_14/SumSum kernel/Regularizer_14/Square:y:0$kernel/Regularizer_14/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/Sum
kernel/Regularizer_14/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_14/mul/xจ
kernel/Regularizer_14/mulMul$kernel/Regularizer_14/mul/x:output:0"kernel/Regularizer_14/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/mulน
+kernel/Regularizer_15/Square/ReadVariableOpReadVariableOpexpand3x3fire6_266544*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_15/Square/ReadVariableOpญ
kernel/Regularizer_15/SquareSquare3kernel/Regularizer_15/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_15/Square
kernel/Regularizer_15/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_15/Constฆ
kernel/Regularizer_15/SumSum kernel/Regularizer_15/Square:y:0$kernel/Regularizer_15/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/Sum
kernel/Regularizer_15/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_15/mul/xจ
kernel/Regularizer_15/mulMul$kernel/Regularizer_15/mul/x:output:0"kernel/Regularizer_15/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/mulท
+kernel/Regularizer_16/Square/ReadVariableOpReadVariableOpsqueezefire7_266550*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_16/Square/ReadVariableOpญ
kernel/Regularizer_16/SquareSquare3kernel/Regularizer_16/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_16/Square
kernel/Regularizer_16/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_16/Constฆ
kernel/Regularizer_16/SumSum kernel/Regularizer_16/Square:y:0$kernel/Regularizer_16/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/Sum
kernel/Regularizer_16/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_16/mul/xจ
kernel/Regularizer_16/mulMul$kernel/Regularizer_16/mul/x:output:0"kernel/Regularizer_16/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/mulน
+kernel/Regularizer_17/Square/ReadVariableOpReadVariableOpexpand1x1fire7_266555*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_17/Square/ReadVariableOpญ
kernel/Regularizer_17/SquareSquare3kernel/Regularizer_17/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_17/Square
kernel/Regularizer_17/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_17/Constฆ
kernel/Regularizer_17/SumSum kernel/Regularizer_17/Square:y:0$kernel/Regularizer_17/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/Sum
kernel/Regularizer_17/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_17/mul/xจ
kernel/Regularizer_17/mulMul$kernel/Regularizer_17/mul/x:output:0"kernel/Regularizer_17/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/mulน
+kernel/Regularizer_18/Square/ReadVariableOpReadVariableOpexpand3x3fire7_266560*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_18/Square/ReadVariableOpญ
kernel/Regularizer_18/SquareSquare3kernel/Regularizer_18/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_18/Square
kernel/Regularizer_18/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_18/Constฆ
kernel/Regularizer_18/SumSum kernel/Regularizer_18/Square:y:0$kernel/Regularizer_18/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/Sum
kernel/Regularizer_18/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_18/mul/xจ
kernel/Regularizer_18/mulMul$kernel/Regularizer_18/mul/x:output:0"kernel/Regularizer_18/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/mulท
+kernel/Regularizer_19/Square/ReadVariableOpReadVariableOpsqueezefire8_266566*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_19/Square/ReadVariableOpญ
kernel/Regularizer_19/SquareSquare3kernel/Regularizer_19/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_19/Square
kernel/Regularizer_19/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_19/Constฆ
kernel/Regularizer_19/SumSum kernel/Regularizer_19/Square:y:0$kernel/Regularizer_19/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/Sum
kernel/Regularizer_19/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_19/mul/xจ
kernel/Regularizer_19/mulMul$kernel/Regularizer_19/mul/x:output:0"kernel/Regularizer_19/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/mulน
+kernel/Regularizer_20/Square/ReadVariableOpReadVariableOpexpand1x1fire8_266571*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_20/Square/ReadVariableOpญ
kernel/Regularizer_20/SquareSquare3kernel/Regularizer_20/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_20/Square
kernel/Regularizer_20/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_20/Constฆ
kernel/Regularizer_20/SumSum kernel/Regularizer_20/Square:y:0$kernel/Regularizer_20/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/Sum
kernel/Regularizer_20/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_20/mul/xจ
kernel/Regularizer_20/mulMul$kernel/Regularizer_20/mul/x:output:0"kernel/Regularizer_20/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/mulน
+kernel/Regularizer_21/Square/ReadVariableOpReadVariableOpexpand3x3fire8_266576*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_21/Square/ReadVariableOpญ
kernel/Regularizer_21/SquareSquare3kernel/Regularizer_21/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_21/Square
kernel/Regularizer_21/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_21/Constฆ
kernel/Regularizer_21/SumSum kernel/Regularizer_21/Square:y:0$kernel/Regularizer_21/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/Sum
kernel/Regularizer_21/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_21/mul/xจ
kernel/Regularizer_21/mulMul$kernel/Regularizer_21/mul/x:output:0"kernel/Regularizer_21/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/mulท
+kernel/Regularizer_22/Square/ReadVariableOpReadVariableOpsqueezefire9_266583*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_22/Square/ReadVariableOpญ
kernel/Regularizer_22/SquareSquare3kernel/Regularizer_22/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_22/Square
kernel/Regularizer_22/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_22/Constฆ
kernel/Regularizer_22/SumSum kernel/Regularizer_22/Square:y:0$kernel/Regularizer_22/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/Sum
kernel/Regularizer_22/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_22/mul/xจ
kernel/Regularizer_22/mulMul$kernel/Regularizer_22/mul/x:output:0"kernel/Regularizer_22/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/mulน
+kernel/Regularizer_23/Square/ReadVariableOpReadVariableOpexpand1x1fire9_266588*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_23/Square/ReadVariableOpญ
kernel/Regularizer_23/SquareSquare3kernel/Regularizer_23/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_23/Square
kernel/Regularizer_23/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_23/Constฆ
kernel/Regularizer_23/SumSum kernel/Regularizer_23/Square:y:0$kernel/Regularizer_23/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/Sum
kernel/Regularizer_23/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_23/mul/xจ
kernel/Regularizer_23/mulMul$kernel/Regularizer_23/mul/x:output:0"kernel/Regularizer_23/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/mulน
+kernel/Regularizer_24/Square/ReadVariableOpReadVariableOpexpand3x3fire9_266593*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_24/Square/ReadVariableOpญ
kernel/Regularizer_24/SquareSquare3kernel/Regularizer_24/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_24/Square
kernel/Regularizer_24/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_24/Constฆ
kernel/Regularizer_24/SumSum kernel/Regularizer_24/Square:y:0$kernel/Regularizer_24/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/Sum
kernel/Regularizer_24/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_24/mul/xจ
kernel/Regularizer_24/mulMul$kernel/Regularizer_24/mul/x:output:0"kernel/Regularizer_24/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/mulฒ	
IdentityIdentity+DenseFinal/StatefulPartitionedCall:output:0!^Conv2D_1/StatefulPartitionedCall#^DenseFinal/StatefulPartitionedCall!^Dropout9/StatefulPartitionedCall'^Expand1x1Fire2/StatefulPartitionedCall'^Expand1x1Fire3/StatefulPartitionedCall'^Expand1x1Fire4/StatefulPartitionedCall'^Expand1x1Fire5/StatefulPartitionedCall'^Expand1x1Fire6/StatefulPartitionedCall'^Expand1x1Fire7/StatefulPartitionedCall'^Expand1x1Fire8/StatefulPartitionedCall'^Expand1x1Fire9/StatefulPartitionedCall'^Expand3x3Fire2/StatefulPartitionedCall'^Expand3x3Fire3/StatefulPartitionedCall'^Expand3x3Fire4/StatefulPartitionedCall'^Expand3x3Fire5/StatefulPartitionedCall'^Expand3x3Fire6/StatefulPartitionedCall'^Expand3x3Fire7/StatefulPartitionedCall'^Expand3x3Fire8/StatefulPartitionedCall'^Expand3x3Fire9/StatefulPartitionedCall%^SqueezeFire2/StatefulPartitionedCall%^SqueezeFire3/StatefulPartitionedCall%^SqueezeFire4/StatefulPartitionedCall%^SqueezeFire5/StatefulPartitionedCall%^SqueezeFire6/StatefulPartitionedCall%^SqueezeFire7/StatefulPartitionedCall%^SqueezeFire8/StatefulPartitionedCall%^SqueezeFire9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2H
"DenseFinal/StatefulPartitionedCall"DenseFinal/StatefulPartitionedCall2D
 Dropout9/StatefulPartitionedCall Dropout9/StatefulPartitionedCall2P
&Expand1x1Fire2/StatefulPartitionedCall&Expand1x1Fire2/StatefulPartitionedCall2P
&Expand1x1Fire3/StatefulPartitionedCall&Expand1x1Fire3/StatefulPartitionedCall2P
&Expand1x1Fire4/StatefulPartitionedCall&Expand1x1Fire4/StatefulPartitionedCall2P
&Expand1x1Fire5/StatefulPartitionedCall&Expand1x1Fire5/StatefulPartitionedCall2P
&Expand1x1Fire6/StatefulPartitionedCall&Expand1x1Fire6/StatefulPartitionedCall2P
&Expand1x1Fire7/StatefulPartitionedCall&Expand1x1Fire7/StatefulPartitionedCall2P
&Expand1x1Fire8/StatefulPartitionedCall&Expand1x1Fire8/StatefulPartitionedCall2P
&Expand1x1Fire9/StatefulPartitionedCall&Expand1x1Fire9/StatefulPartitionedCall2P
&Expand3x3Fire2/StatefulPartitionedCall&Expand3x3Fire2/StatefulPartitionedCall2P
&Expand3x3Fire3/StatefulPartitionedCall&Expand3x3Fire3/StatefulPartitionedCall2P
&Expand3x3Fire4/StatefulPartitionedCall&Expand3x3Fire4/StatefulPartitionedCall2P
&Expand3x3Fire5/StatefulPartitionedCall&Expand3x3Fire5/StatefulPartitionedCall2P
&Expand3x3Fire6/StatefulPartitionedCall&Expand3x3Fire6/StatefulPartitionedCall2P
&Expand3x3Fire7/StatefulPartitionedCall&Expand3x3Fire7/StatefulPartitionedCall2P
&Expand3x3Fire8/StatefulPartitionedCall&Expand3x3Fire8/StatefulPartitionedCall2P
&Expand3x3Fire9/StatefulPartitionedCall&Expand3x3Fire9/StatefulPartitionedCall2L
$SqueezeFire2/StatefulPartitionedCall$SqueezeFire2/StatefulPartitionedCall2L
$SqueezeFire3/StatefulPartitionedCall$SqueezeFire3/StatefulPartitionedCall2L
$SqueezeFire4/StatefulPartitionedCall$SqueezeFire4/StatefulPartitionedCall2L
$SqueezeFire5/StatefulPartitionedCall$SqueezeFire5/StatefulPartitionedCall2L
$SqueezeFire6/StatefulPartitionedCall$SqueezeFire6/StatefulPartitionedCall2L
$SqueezeFire7/StatefulPartitionedCall$SqueezeFire7/StatefulPartitionedCall2L
$SqueezeFire8/StatefulPartitionedCall$SqueezeFire8/StatefulPartitionedCall2L
$SqueezeFire9/StatefulPartitionedCall$SqueezeFire9/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
๏
t
H__inference_Concatenate3_layer_call_and_return_conditional_losses_268722
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????77@:?????????77@:Y U
/
_output_shapes
:?????????77@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????77@
"
_user_specified_name
inputs/1
ฮ๔

"__inference__traced_restore_270108
file_prefix$
 assignvariableop_conv2d_1_kernel$
 assignvariableop_1_conv2d_1_bias*
&assignvariableop_2_squeezefire2_kernel(
$assignvariableop_3_squeezefire2_bias,
(assignvariableop_4_expand1x1fire2_kernel*
&assignvariableop_5_expand1x1fire2_bias,
(assignvariableop_6_expand3x3fire2_kernel*
&assignvariableop_7_expand3x3fire2_bias*
&assignvariableop_8_squeezefire3_kernel(
$assignvariableop_9_squeezefire3_bias-
)assignvariableop_10_expand1x1fire3_kernel+
'assignvariableop_11_expand1x1fire3_bias-
)assignvariableop_12_expand3x3fire3_kernel+
'assignvariableop_13_expand3x3fire3_bias+
'assignvariableop_14_squeezefire4_kernel)
%assignvariableop_15_squeezefire4_bias-
)assignvariableop_16_expand1x1fire4_kernel+
'assignvariableop_17_expand1x1fire4_bias-
)assignvariableop_18_expand3x3fire4_kernel+
'assignvariableop_19_expand3x3fire4_bias+
'assignvariableop_20_squeezefire5_kernel)
%assignvariableop_21_squeezefire5_bias-
)assignvariableop_22_expand1x1fire5_kernel+
'assignvariableop_23_expand1x1fire5_bias-
)assignvariableop_24_expand3x3fire5_kernel+
'assignvariableop_25_expand3x3fire5_bias+
'assignvariableop_26_squeezefire6_kernel)
%assignvariableop_27_squeezefire6_bias-
)assignvariableop_28_expand1x1fire6_kernel+
'assignvariableop_29_expand1x1fire6_bias-
)assignvariableop_30_expand3x3fire6_kernel+
'assignvariableop_31_expand3x3fire6_bias+
'assignvariableop_32_squeezefire7_kernel)
%assignvariableop_33_squeezefire7_bias-
)assignvariableop_34_expand1x1fire7_kernel+
'assignvariableop_35_expand1x1fire7_bias-
)assignvariableop_36_expand3x3fire7_kernel+
'assignvariableop_37_expand3x3fire7_bias+
'assignvariableop_38_squeezefire8_kernel)
%assignvariableop_39_squeezefire8_bias-
)assignvariableop_40_expand1x1fire8_kernel+
'assignvariableop_41_expand1x1fire8_bias-
)assignvariableop_42_expand3x3fire8_kernel+
'assignvariableop_43_expand3x3fire8_bias+
'assignvariableop_44_squeezefire9_kernel)
%assignvariableop_45_squeezefire9_bias-
)assignvariableop_46_expand1x1fire9_kernel+
'assignvariableop_47_expand1x1fire9_bias-
)assignvariableop_48_expand3x3fire9_kernel+
'assignvariableop_49_expand3x3fire9_bias)
%assignvariableop_50_densefinal_kernel'
#assignvariableop_51_densefinal_bias 
assignvariableop_52_sgd_iter!
assignvariableop_53_sgd_decay)
%assignvariableop_54_sgd_learning_rate$
 assignvariableop_55_sgd_momentum
assignvariableop_56_total
assignvariableop_57_count
assignvariableop_58_total_1
assignvariableop_59_count_1
identity_61ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_33ขAssignVariableOp_34ขAssignVariableOp_35ขAssignVariableOp_36ขAssignVariableOp_37ขAssignVariableOp_38ขAssignVariableOp_39ขAssignVariableOp_4ขAssignVariableOp_40ขAssignVariableOp_41ขAssignVariableOp_42ขAssignVariableOp_43ขAssignVariableOp_44ขAssignVariableOp_45ขAssignVariableOp_46ขAssignVariableOp_47ขAssignVariableOp_48ขAssignVariableOp_49ขAssignVariableOp_5ขAssignVariableOp_50ขAssignVariableOp_51ขAssignVariableOp_52ขAssignVariableOp_53ขAssignVariableOp_54ขAssignVariableOp_55ขAssignVariableOp_56ขAssignVariableOp_57ขAssignVariableOp_58ขAssignVariableOp_59ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices฿
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes๗
๔:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ฅ
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ซ
AssignVariableOp_2AssignVariableOp&assignvariableop_2_squeezefire2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ฉ
AssignVariableOp_3AssignVariableOp$assignvariableop_3_squeezefire2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ญ
AssignVariableOp_4AssignVariableOp(assignvariableop_4_expand1x1fire2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ซ
AssignVariableOp_5AssignVariableOp&assignvariableop_5_expand1x1fire2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ญ
AssignVariableOp_6AssignVariableOp(assignvariableop_6_expand3x3fire2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ซ
AssignVariableOp_7AssignVariableOp&assignvariableop_7_expand3x3fire2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ซ
AssignVariableOp_8AssignVariableOp&assignvariableop_8_squeezefire3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ฉ
AssignVariableOp_9AssignVariableOp$assignvariableop_9_squeezefire3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ฑ
AssignVariableOp_10AssignVariableOp)assignvariableop_10_expand1x1fire3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ฏ
AssignVariableOp_11AssignVariableOp'assignvariableop_11_expand1x1fire3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ฑ
AssignVariableOp_12AssignVariableOp)assignvariableop_12_expand3x3fire3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ฏ
AssignVariableOp_13AssignVariableOp'assignvariableop_13_expand3x3fire3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ฏ
AssignVariableOp_14AssignVariableOp'assignvariableop_14_squeezefire4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ญ
AssignVariableOp_15AssignVariableOp%assignvariableop_15_squeezefire4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ฑ
AssignVariableOp_16AssignVariableOp)assignvariableop_16_expand1x1fire4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ฏ
AssignVariableOp_17AssignVariableOp'assignvariableop_17_expand1x1fire4_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ฑ
AssignVariableOp_18AssignVariableOp)assignvariableop_18_expand3x3fire4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ฏ
AssignVariableOp_19AssignVariableOp'assignvariableop_19_expand3x3fire4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ฏ
AssignVariableOp_20AssignVariableOp'assignvariableop_20_squeezefire5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ญ
AssignVariableOp_21AssignVariableOp%assignvariableop_21_squeezefire5_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ฑ
AssignVariableOp_22AssignVariableOp)assignvariableop_22_expand1x1fire5_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ฏ
AssignVariableOp_23AssignVariableOp'assignvariableop_23_expand1x1fire5_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ฑ
AssignVariableOp_24AssignVariableOp)assignvariableop_24_expand3x3fire5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ฏ
AssignVariableOp_25AssignVariableOp'assignvariableop_25_expand3x3fire5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ฏ
AssignVariableOp_26AssignVariableOp'assignvariableop_26_squeezefire6_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ญ
AssignVariableOp_27AssignVariableOp%assignvariableop_27_squeezefire6_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ฑ
AssignVariableOp_28AssignVariableOp)assignvariableop_28_expand1x1fire6_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ฏ
AssignVariableOp_29AssignVariableOp'assignvariableop_29_expand1x1fire6_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ฑ
AssignVariableOp_30AssignVariableOp)assignvariableop_30_expand3x3fire6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ฏ
AssignVariableOp_31AssignVariableOp'assignvariableop_31_expand3x3fire6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ฏ
AssignVariableOp_32AssignVariableOp'assignvariableop_32_squeezefire7_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ญ
AssignVariableOp_33AssignVariableOp%assignvariableop_33_squeezefire7_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ฑ
AssignVariableOp_34AssignVariableOp)assignvariableop_34_expand1x1fire7_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ฏ
AssignVariableOp_35AssignVariableOp'assignvariableop_35_expand1x1fire7_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ฑ
AssignVariableOp_36AssignVariableOp)assignvariableop_36_expand3x3fire7_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ฏ
AssignVariableOp_37AssignVariableOp'assignvariableop_37_expand3x3fire7_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ฏ
AssignVariableOp_38AssignVariableOp'assignvariableop_38_squeezefire8_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39ญ
AssignVariableOp_39AssignVariableOp%assignvariableop_39_squeezefire8_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ฑ
AssignVariableOp_40AssignVariableOp)assignvariableop_40_expand1x1fire8_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41ฏ
AssignVariableOp_41AssignVariableOp'assignvariableop_41_expand1x1fire8_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ฑ
AssignVariableOp_42AssignVariableOp)assignvariableop_42_expand3x3fire8_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43ฏ
AssignVariableOp_43AssignVariableOp'assignvariableop_43_expand3x3fire8_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44ฏ
AssignVariableOp_44AssignVariableOp'assignvariableop_44_squeezefire9_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45ญ
AssignVariableOp_45AssignVariableOp%assignvariableop_45_squeezefire9_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46ฑ
AssignVariableOp_46AssignVariableOp)assignvariableop_46_expand1x1fire9_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47ฏ
AssignVariableOp_47AssignVariableOp'assignvariableop_47_expand1x1fire9_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ฑ
AssignVariableOp_48AssignVariableOp)assignvariableop_48_expand3x3fire9_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49ฏ
AssignVariableOp_49AssignVariableOp'assignvariableop_49_expand3x3fire9_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50ญ
AssignVariableOp_50AssignVariableOp%assignvariableop_50_densefinal_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51ซ
AssignVariableOp_51AssignVariableOp#assignvariableop_51_densefinal_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_52ค
AssignVariableOp_52AssignVariableOpassignvariableop_52_sgd_iterIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53ฅ
AssignVariableOp_53AssignVariableOpassignvariableop_53_sgd_decayIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54ญ
AssignVariableOp_54AssignVariableOp%assignvariableop_54_sgd_learning_rateIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55จ
AssignVariableOp_55AssignVariableOp assignvariableop_55_sgd_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56ก
AssignVariableOp_56AssignVariableOpassignvariableop_56_totalIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ก
AssignVariableOp_57AssignVariableOpassignvariableop_57_countIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58ฃ
AssignVariableOp_58AssignVariableOpassignvariableop_58_total_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59ฃ
AssignVariableOp_59AssignVariableOpassignvariableop_59_count_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_599
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_60๙

Identity_61IdentityIdentity_60:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_61"#
identity_61Identity_61:output:0*
_input_shapes๕
๒: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
กs
ส
__inference__traced_save_269918
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop2
.savev2_squeezefire2_kernel_read_readvariableop0
,savev2_squeezefire2_bias_read_readvariableop4
0savev2_expand1x1fire2_kernel_read_readvariableop2
.savev2_expand1x1fire2_bias_read_readvariableop4
0savev2_expand3x3fire2_kernel_read_readvariableop2
.savev2_expand3x3fire2_bias_read_readvariableop2
.savev2_squeezefire3_kernel_read_readvariableop0
,savev2_squeezefire3_bias_read_readvariableop4
0savev2_expand1x1fire3_kernel_read_readvariableop2
.savev2_expand1x1fire3_bias_read_readvariableop4
0savev2_expand3x3fire3_kernel_read_readvariableop2
.savev2_expand3x3fire3_bias_read_readvariableop2
.savev2_squeezefire4_kernel_read_readvariableop0
,savev2_squeezefire4_bias_read_readvariableop4
0savev2_expand1x1fire4_kernel_read_readvariableop2
.savev2_expand1x1fire4_bias_read_readvariableop4
0savev2_expand3x3fire4_kernel_read_readvariableop2
.savev2_expand3x3fire4_bias_read_readvariableop2
.savev2_squeezefire5_kernel_read_readvariableop0
,savev2_squeezefire5_bias_read_readvariableop4
0savev2_expand1x1fire5_kernel_read_readvariableop2
.savev2_expand1x1fire5_bias_read_readvariableop4
0savev2_expand3x3fire5_kernel_read_readvariableop2
.savev2_expand3x3fire5_bias_read_readvariableop2
.savev2_squeezefire6_kernel_read_readvariableop0
,savev2_squeezefire6_bias_read_readvariableop4
0savev2_expand1x1fire6_kernel_read_readvariableop2
.savev2_expand1x1fire6_bias_read_readvariableop4
0savev2_expand3x3fire6_kernel_read_readvariableop2
.savev2_expand3x3fire6_bias_read_readvariableop2
.savev2_squeezefire7_kernel_read_readvariableop0
,savev2_squeezefire7_bias_read_readvariableop4
0savev2_expand1x1fire7_kernel_read_readvariableop2
.savev2_expand1x1fire7_bias_read_readvariableop4
0savev2_expand3x3fire7_kernel_read_readvariableop2
.savev2_expand3x3fire7_bias_read_readvariableop2
.savev2_squeezefire8_kernel_read_readvariableop0
,savev2_squeezefire8_bias_read_readvariableop4
0savev2_expand1x1fire8_kernel_read_readvariableop2
.savev2_expand1x1fire8_bias_read_readvariableop4
0savev2_expand3x3fire8_kernel_read_readvariableop2
.savev2_expand3x3fire8_bias_read_readvariableop2
.savev2_squeezefire9_kernel_read_readvariableop0
,savev2_squeezefire9_bias_read_readvariableop4
0savev2_expand1x1fire9_kernel_read_readvariableop2
.savev2_expand1x1fire9_bias_read_readvariableop4
0savev2_expand3x3fire9_kernel_read_readvariableop2
.savev2_expand3x3fire9_bias_read_readvariableop0
,savev2_densefinal_kernel_read_readvariableop.
*savev2_densefinal_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ac46d150c1ac4dab8dace353216f2852/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardฆ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesไ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop.savev2_squeezefire2_kernel_read_readvariableop,savev2_squeezefire2_bias_read_readvariableop0savev2_expand1x1fire2_kernel_read_readvariableop.savev2_expand1x1fire2_bias_read_readvariableop0savev2_expand3x3fire2_kernel_read_readvariableop.savev2_expand3x3fire2_bias_read_readvariableop.savev2_squeezefire3_kernel_read_readvariableop,savev2_squeezefire3_bias_read_readvariableop0savev2_expand1x1fire3_kernel_read_readvariableop.savev2_expand1x1fire3_bias_read_readvariableop0savev2_expand3x3fire3_kernel_read_readvariableop.savev2_expand3x3fire3_bias_read_readvariableop.savev2_squeezefire4_kernel_read_readvariableop,savev2_squeezefire4_bias_read_readvariableop0savev2_expand1x1fire4_kernel_read_readvariableop.savev2_expand1x1fire4_bias_read_readvariableop0savev2_expand3x3fire4_kernel_read_readvariableop.savev2_expand3x3fire4_bias_read_readvariableop.savev2_squeezefire5_kernel_read_readvariableop,savev2_squeezefire5_bias_read_readvariableop0savev2_expand1x1fire5_kernel_read_readvariableop.savev2_expand1x1fire5_bias_read_readvariableop0savev2_expand3x3fire5_kernel_read_readvariableop.savev2_expand3x3fire5_bias_read_readvariableop.savev2_squeezefire6_kernel_read_readvariableop,savev2_squeezefire6_bias_read_readvariableop0savev2_expand1x1fire6_kernel_read_readvariableop.savev2_expand1x1fire6_bias_read_readvariableop0savev2_expand3x3fire6_kernel_read_readvariableop.savev2_expand3x3fire6_bias_read_readvariableop.savev2_squeezefire7_kernel_read_readvariableop,savev2_squeezefire7_bias_read_readvariableop0savev2_expand1x1fire7_kernel_read_readvariableop.savev2_expand1x1fire7_bias_read_readvariableop0savev2_expand3x3fire7_kernel_read_readvariableop.savev2_expand3x3fire7_bias_read_readvariableop.savev2_squeezefire8_kernel_read_readvariableop,savev2_squeezefire8_bias_read_readvariableop0savev2_expand1x1fire8_kernel_read_readvariableop.savev2_expand1x1fire8_bias_read_readvariableop0savev2_expand3x3fire8_kernel_read_readvariableop.savev2_expand3x3fire8_bias_read_readvariableop.savev2_squeezefire9_kernel_read_readvariableop,savev2_squeezefire9_bias_read_readvariableop0savev2_expand1x1fire9_kernel_read_readvariableop.savev2_expand1x1fire9_bias_read_readvariableop0savev2_expand3x3fire9_kernel_read_readvariableop.savev2_expand3x3fire9_bias_read_readvariableop,savev2_densefinal_kernel_read_readvariableop*savev2_densefinal_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *K
dtypesA
?2=	2
SaveV2บ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesก
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ฒ
_input_shapes?
: :@:@:@::@:@:@:@:::@:@:@:@: : : :: :: : : :: ::0:0:0ภ:ภ:0ภ:ภ:0:0:0ภ:ภ:0ภ:ภ:@:@:@::@::@:@:@::@::
ค:: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@:-	)
'
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: :!

_output_shapes	
::-)
'
_output_shapes
: :!

_output_shapes	
::-)
'
_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: :!

_output_shapes	
::-)
'
_output_shapes
: :!

_output_shapes	
::-)
'
_output_shapes
:0: 

_output_shapes
:0:-)
'
_output_shapes
:0ภ:!

_output_shapes	
:ภ:-)
'
_output_shapes
:0ภ:! 

_output_shapes	
:ภ:-!)
'
_output_shapes
:0: "

_output_shapes
:0:-#)
'
_output_shapes
:0ภ:!$

_output_shapes	
:ภ:-%)
'
_output_shapes
:0ภ:!&

_output_shapes	
:ภ:-')
'
_output_shapes
:@: (

_output_shapes
:@:-))
'
_output_shapes
:@:!*

_output_shapes	
::-+)
'
_output_shapes
:@:!,

_output_shapes	
::--)
'
_output_shapes
:@: .

_output_shapes
:@:-/)
'
_output_shapes
:@:!0

_output_shapes	
::-1)
'
_output_shapes
:@:!2

_output_shapes	
::&3"
 
_output_shapes
:
ค: 4

_output_shapes
::5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: 
ย
ฒ
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_269142

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs


/__inference_Expand1x1Fire3_layer_call_fn_268683

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_2651812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_265379

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_13_2695945
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:0*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ย
ฒ
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_265906

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
๋
r
H__inference_Concatenate6_layer_call_and_return_conditional_losses_265583

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????ภ:?????????ภ:X T
0
_output_shapes
:?????????ภ
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????ภ
 
_user_specified_nameinputs
ิ
Y
-__inference_Concatenate2_layer_call_fn_268619
inputs_0
inputs_1
identity฿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate2_layer_call_and_return_conditional_losses_2651222
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????77@:?????????77@:Y U
/
_output_shapes
:?????????77@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????77@
"
_user_specified_name
inputs/1
ฝ
ฐ
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_268642

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????772
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????77:::X T
0
_output_shapes
:?????????77
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_11_2695725
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ฟ	
f
__inference_loss_fn_4_2694955
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ย
ฒ
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_269110

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs


/__inference_Expand3x3Fire2_layer_call_fn_268606

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_2650992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
็
r
H__inference_Concatenate3_layer_call_and_return_conditional_losses_265237

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????77@:?????????77@:W S
/
_output_shapes
:?????????77@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????77@
 
_user_specified_nameinputs
๚
`
D__inference_MaxPool1_layer_call_and_return_conditional_losses_264948

inputs
identityญ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_265609

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????02
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

~
)__inference_Conv2D_1_layer_call_fn_268510

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_2649992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????pp@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????เเ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_16_2696275
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:0*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:


/__inference_Expand3x3Fire9_layer_call_fn_269369

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_2659062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ฝ	
f
__inference_loss_fn_5_2695065
1kernel_regularizer_square_readvariableop_resource
identityฮ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:


/__inference_Expand1x1Fire2_layer_call_fn_268574

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_2650662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_10_2695615
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ภ	
g
__inference_loss_fn_12_2695835
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
บ
ฒ
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_265181

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77:::W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
ุ
Y
-__inference_Concatenate9_layer_call_fn_269382
inputs_0
inputs_1
identity฿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate9_layer_call_and_return_conditional_losses_2659292
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ก
ก
5__inference_SqueezeNet_Preloaded_layer_call_fn_268369

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identityขStatefulPartitionedCallถ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_2667572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
๋
r
H__inference_Concatenate4_layer_call_and_return_conditional_losses_265352

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????77:?????????77:X T
0
_output_shapes
:?????????77
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????77
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_265412

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
๚
`
D__inference_MaxPool4_layer_call_and_return_conditional_losses_264960

inputs
identityญ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_265757

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ธ
ฌ
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_268501

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????pp@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????pp@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:?????????เเ:::Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
๋
r
H__inference_Concatenate5_layer_call_and_return_conditional_losses_265468

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs
?ร
ฑ
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_267902

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource/
+squeezefire2_conv2d_readvariableop_resource0
,squeezefire2_biasadd_readvariableop_resource1
-expand1x1fire2_conv2d_readvariableop_resource2
.expand1x1fire2_biasadd_readvariableop_resource1
-expand3x3fire2_conv2d_readvariableop_resource2
.expand3x3fire2_biasadd_readvariableop_resource/
+squeezefire3_conv2d_readvariableop_resource0
,squeezefire3_biasadd_readvariableop_resource1
-expand1x1fire3_conv2d_readvariableop_resource2
.expand1x1fire3_biasadd_readvariableop_resource1
-expand3x3fire3_conv2d_readvariableop_resource2
.expand3x3fire3_biasadd_readvariableop_resource/
+squeezefire4_conv2d_readvariableop_resource0
,squeezefire4_biasadd_readvariableop_resource1
-expand1x1fire4_conv2d_readvariableop_resource2
.expand1x1fire4_biasadd_readvariableop_resource1
-expand3x3fire4_conv2d_readvariableop_resource2
.expand3x3fire4_biasadd_readvariableop_resource/
+squeezefire5_conv2d_readvariableop_resource0
,squeezefire5_biasadd_readvariableop_resource1
-expand1x1fire5_conv2d_readvariableop_resource2
.expand1x1fire5_biasadd_readvariableop_resource1
-expand3x3fire5_conv2d_readvariableop_resource2
.expand3x3fire5_biasadd_readvariableop_resource/
+squeezefire6_conv2d_readvariableop_resource0
,squeezefire6_biasadd_readvariableop_resource1
-expand1x1fire6_conv2d_readvariableop_resource2
.expand1x1fire6_biasadd_readvariableop_resource1
-expand3x3fire6_conv2d_readvariableop_resource2
.expand3x3fire6_biasadd_readvariableop_resource/
+squeezefire7_conv2d_readvariableop_resource0
,squeezefire7_biasadd_readvariableop_resource1
-expand1x1fire7_conv2d_readvariableop_resource2
.expand1x1fire7_biasadd_readvariableop_resource1
-expand3x3fire7_conv2d_readvariableop_resource2
.expand3x3fire7_biasadd_readvariableop_resource/
+squeezefire8_conv2d_readvariableop_resource0
,squeezefire8_biasadd_readvariableop_resource1
-expand1x1fire8_conv2d_readvariableop_resource2
.expand1x1fire8_biasadd_readvariableop_resource1
-expand3x3fire8_conv2d_readvariableop_resource2
.expand3x3fire8_biasadd_readvariableop_resource/
+squeezefire9_conv2d_readvariableop_resource0
,squeezefire9_biasadd_readvariableop_resource1
-expand1x1fire9_conv2d_readvariableop_resource2
.expand1x1fire9_biasadd_readvariableop_resource1
-expand3x3fire9_conv2d_readvariableop_resource2
.expand3x3fire9_biasadd_readvariableop_resource-
)densefinal_matmul_readvariableop_resource.
*densefinal_biasadd_readvariableop_resource
identityฐ
Conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
Conv2D_1/Conv2D/ReadVariableOpพ
Conv2D_1/Conv2DConv2Dinputs&Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@*
paddingSAME*
strides
2
Conv2D_1/Conv2Dง
Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
Conv2D_1/BiasAdd/ReadVariableOpฌ
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2D:output:0'Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@2
Conv2D_1/BiasAdd{
Conv2D_1/ReluReluConv2D_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????pp@2
Conv2D_1/Reluน
MaxPool1/MaxPoolMaxPoolConv2D_1/Relu:activations:0*/
_output_shapes
:?????????77@*
ksize
*
paddingVALID*
strides
2
MaxPool1/MaxPoolผ
"SqueezeFire2/Conv2D/ReadVariableOpReadVariableOp+squeezefire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"SqueezeFire2/Conv2D/ReadVariableOp?
SqueezeFire2/Conv2DConv2DMaxPool1/MaxPool:output:0*SqueezeFire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2
SqueezeFire2/Conv2Dณ
#SqueezeFire2/BiasAdd/ReadVariableOpReadVariableOp,squeezefire2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#SqueezeFire2/BiasAdd/ReadVariableOpผ
SqueezeFire2/BiasAddBiasAddSqueezeFire2/Conv2D:output:0+SqueezeFire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772
SqueezeFire2/BiasAdd
SqueezeFire2/ReluReluSqueezeFire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????772
SqueezeFire2/Reluย
$Expand1x1Fire2/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Expand1x1Fire2/Conv2D/ReadVariableOp้
Expand1x1Fire2/Conv2DConv2DSqueezeFire2/Relu:activations:0,Expand1x1Fire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Expand1x1Fire2/Conv2Dน
%Expand1x1Fire2/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%Expand1x1Fire2/BiasAdd/ReadVariableOpฤ
Expand1x1Fire2/BiasAddBiasAddExpand1x1Fire2/Conv2D:output:0-Expand1x1Fire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2
Expand1x1Fire2/BiasAdd
Expand1x1Fire2/ReluReluExpand1x1Fire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Expand1x1Fire2/Reluย
$Expand3x3Fire2/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Expand3x3Fire2/Conv2D/ReadVariableOp้
Expand3x3Fire2/Conv2DConv2DSqueezeFire2/Relu:activations:0,Expand3x3Fire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Expand3x3Fire2/Conv2Dน
%Expand3x3Fire2/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%Expand3x3Fire2/BiasAdd/ReadVariableOpฤ
Expand3x3Fire2/BiasAddBiasAddExpand3x3Fire2/Conv2D:output:0-Expand3x3Fire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2
Expand3x3Fire2/BiasAdd
Expand3x3Fire2/ReluReluExpand3x3Fire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Expand3x3Fire2/Reluv
Concatenate2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate2/concat/axisใ
Concatenate2/concatConcatV2!Expand1x1Fire2/Relu:activations:0!Expand3x3Fire2/Relu:activations:0!Concatenate2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
Concatenate2/concatฝ
"SqueezeFire3/Conv2D/ReadVariableOpReadVariableOp+squeezefire3_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02$
"SqueezeFire3/Conv2D/ReadVariableOpเ
SqueezeFire3/Conv2DConv2DConcatenate2/concat:output:0*SqueezeFire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2
SqueezeFire3/Conv2Dณ
#SqueezeFire3/BiasAdd/ReadVariableOpReadVariableOp,squeezefire3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#SqueezeFire3/BiasAdd/ReadVariableOpผ
SqueezeFire3/BiasAddBiasAddSqueezeFire3/Conv2D:output:0+SqueezeFire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772
SqueezeFire3/BiasAdd
SqueezeFire3/ReluReluSqueezeFire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????772
SqueezeFire3/Reluย
$Expand1x1Fire3/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Expand1x1Fire3/Conv2D/ReadVariableOp้
Expand1x1Fire3/Conv2DConv2DSqueezeFire3/Relu:activations:0,Expand1x1Fire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Expand1x1Fire3/Conv2Dน
%Expand1x1Fire3/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%Expand1x1Fire3/BiasAdd/ReadVariableOpฤ
Expand1x1Fire3/BiasAddBiasAddExpand1x1Fire3/Conv2D:output:0-Expand1x1Fire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2
Expand1x1Fire3/BiasAdd
Expand1x1Fire3/ReluReluExpand1x1Fire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Expand1x1Fire3/Reluย
$Expand3x3Fire3/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Expand3x3Fire3/Conv2D/ReadVariableOp้
Expand3x3Fire3/Conv2DConv2DSqueezeFire3/Relu:activations:0,Expand3x3Fire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Expand3x3Fire3/Conv2Dน
%Expand3x3Fire3/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%Expand3x3Fire3/BiasAdd/ReadVariableOpฤ
Expand3x3Fire3/BiasAddBiasAddExpand3x3Fire3/Conv2D:output:0-Expand3x3Fire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2
Expand3x3Fire3/BiasAdd
Expand3x3Fire3/ReluReluExpand3x3Fire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Expand3x3Fire3/Reluv
Concatenate3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate3/concat/axisใ
Concatenate3/concatConcatV2!Expand1x1Fire3/Relu:activations:0!Expand3x3Fire3/Relu:activations:0!Concatenate3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
Concatenate3/concatฝ
"SqueezeFire4/Conv2D/ReadVariableOpReadVariableOp+squeezefire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02$
"SqueezeFire4/Conv2D/ReadVariableOpเ
SqueezeFire4/Conv2DConv2DConcatenate3/concat:output:0*SqueezeFire4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 *
paddingSAME*
strides
2
SqueezeFire4/Conv2Dณ
#SqueezeFire4/BiasAdd/ReadVariableOpReadVariableOp,squeezefire4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#SqueezeFire4/BiasAdd/ReadVariableOpผ
SqueezeFire4/BiasAddBiasAddSqueezeFire4/Conv2D:output:0+SqueezeFire4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 2
SqueezeFire4/BiasAdd
SqueezeFire4/ReluReluSqueezeFire4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77 2
SqueezeFire4/Reluร
$Expand1x1Fire4/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02&
$Expand1x1Fire4/Conv2D/ReadVariableOp๊
Expand1x1Fire4/Conv2DConv2DSqueezeFire4/Relu:activations:0,Expand1x1Fire4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2
Expand1x1Fire4/Conv2Dบ
%Expand1x1Fire4/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand1x1Fire4/BiasAdd/ReadVariableOpล
Expand1x1Fire4/BiasAddBiasAddExpand1x1Fire4/Conv2D:output:0-Expand1x1Fire4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772
Expand1x1Fire4/BiasAdd
Expand1x1Fire4/ReluReluExpand1x1Fire4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????772
Expand1x1Fire4/Reluร
$Expand3x3Fire4/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02&
$Expand3x3Fire4/Conv2D/ReadVariableOp๊
Expand3x3Fire4/Conv2DConv2DSqueezeFire4/Relu:activations:0,Expand3x3Fire4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2
Expand3x3Fire4/Conv2Dบ
%Expand3x3Fire4/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand3x3Fire4/BiasAdd/ReadVariableOpล
Expand3x3Fire4/BiasAddBiasAddExpand3x3Fire4/Conv2D:output:0-Expand3x3Fire4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772
Expand3x3Fire4/BiasAdd
Expand3x3Fire4/ReluReluExpand3x3Fire4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????772
Expand3x3Fire4/Reluv
Concatenate4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate4/concat/axisใ
Concatenate4/concatConcatV2!Expand1x1Fire4/Relu:activations:0!Expand3x3Fire4/Relu:activations:0!Concatenate4/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
Concatenate4/concatป
MaxPool4/MaxPoolMaxPoolConcatenate4/concat:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
MaxPool4/MaxPoolฝ
"SqueezeFire5/Conv2D/ReadVariableOpReadVariableOp+squeezefire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02$
"SqueezeFire5/Conv2D/ReadVariableOp?
SqueezeFire5/Conv2DConv2DMaxPool4/MaxPool:output:0*SqueezeFire5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
SqueezeFire5/Conv2Dณ
#SqueezeFire5/BiasAdd/ReadVariableOpReadVariableOp,squeezefire5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#SqueezeFire5/BiasAdd/ReadVariableOpผ
SqueezeFire5/BiasAddBiasAddSqueezeFire5/Conv2D:output:0+SqueezeFire5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
SqueezeFire5/BiasAdd
SqueezeFire5/ReluReluSqueezeFire5/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
SqueezeFire5/Reluร
$Expand1x1Fire5/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02&
$Expand1x1Fire5/Conv2D/ReadVariableOp๊
Expand1x1Fire5/Conv2DConv2DSqueezeFire5/Relu:activations:0,Expand1x1Fire5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand1x1Fire5/Conv2Dบ
%Expand1x1Fire5/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand1x1Fire5/BiasAdd/ReadVariableOpล
Expand1x1Fire5/BiasAddBiasAddExpand1x1Fire5/Conv2D:output:0-Expand1x1Fire5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire5/BiasAdd
Expand1x1Fire5/ReluReluExpand1x1Fire5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire5/Reluร
$Expand3x3Fire5/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02&
$Expand3x3Fire5/Conv2D/ReadVariableOp๊
Expand3x3Fire5/Conv2DConv2DSqueezeFire5/Relu:activations:0,Expand3x3Fire5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand3x3Fire5/Conv2Dบ
%Expand3x3Fire5/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand3x3Fire5/BiasAdd/ReadVariableOpล
Expand3x3Fire5/BiasAddBiasAddExpand3x3Fire5/Conv2D:output:0-Expand3x3Fire5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire5/BiasAdd
Expand3x3Fire5/ReluReluExpand3x3Fire5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire5/Reluv
Concatenate5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate5/concat/axisใ
Concatenate5/concatConcatV2!Expand1x1Fire5/Relu:activations:0!Expand3x3Fire5/Relu:activations:0!Concatenate5/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate5/concatฝ
"SqueezeFire6/Conv2D/ReadVariableOpReadVariableOp+squeezefire6_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02$
"SqueezeFire6/Conv2D/ReadVariableOpเ
SqueezeFire6/Conv2DConv2DConcatenate5/concat:output:0*SqueezeFire6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2
SqueezeFire6/Conv2Dณ
#SqueezeFire6/BiasAdd/ReadVariableOpReadVariableOp,squeezefire6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02%
#SqueezeFire6/BiasAdd/ReadVariableOpผ
SqueezeFire6/BiasAddBiasAddSqueezeFire6/Conv2D:output:0+SqueezeFire6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
SqueezeFire6/BiasAdd
SqueezeFire6/ReluReluSqueezeFire6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
SqueezeFire6/Reluร
$Expand1x1Fire6/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02&
$Expand1x1Fire6/Conv2D/ReadVariableOp๊
Expand1x1Fire6/Conv2DConv2DSqueezeFire6/Relu:activations:0,Expand1x1Fire6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Expand1x1Fire6/Conv2Dบ
%Expand1x1Fire6/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire6_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02'
%Expand1x1Fire6/BiasAdd/ReadVariableOpล
Expand1x1Fire6/BiasAddBiasAddExpand1x1Fire6/Conv2D:output:0-Expand1x1Fire6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2
Expand1x1Fire6/BiasAdd
Expand1x1Fire6/ReluReluExpand1x1Fire6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Expand1x1Fire6/Reluร
$Expand3x3Fire6/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02&
$Expand3x3Fire6/Conv2D/ReadVariableOp๊
Expand3x3Fire6/Conv2DConv2DSqueezeFire6/Relu:activations:0,Expand3x3Fire6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Expand3x3Fire6/Conv2Dบ
%Expand3x3Fire6/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire6_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02'
%Expand3x3Fire6/BiasAdd/ReadVariableOpล
Expand3x3Fire6/BiasAddBiasAddExpand3x3Fire6/Conv2D:output:0-Expand3x3Fire6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2
Expand3x3Fire6/BiasAdd
Expand3x3Fire6/ReluReluExpand3x3Fire6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Expand3x3Fire6/Reluv
Concatenate6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate6/concat/axisใ
Concatenate6/concatConcatV2!Expand1x1Fire6/Relu:activations:0!Expand3x3Fire6/Relu:activations:0!Concatenate6/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate6/concatฝ
"SqueezeFire7/Conv2D/ReadVariableOpReadVariableOp+squeezefire7_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02$
"SqueezeFire7/Conv2D/ReadVariableOpเ
SqueezeFire7/Conv2DConv2DConcatenate6/concat:output:0*SqueezeFire7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2
SqueezeFire7/Conv2Dณ
#SqueezeFire7/BiasAdd/ReadVariableOpReadVariableOp,squeezefire7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02%
#SqueezeFire7/BiasAdd/ReadVariableOpผ
SqueezeFire7/BiasAddBiasAddSqueezeFire7/Conv2D:output:0+SqueezeFire7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
SqueezeFire7/BiasAdd
SqueezeFire7/ReluReluSqueezeFire7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
SqueezeFire7/Reluร
$Expand1x1Fire7/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02&
$Expand1x1Fire7/Conv2D/ReadVariableOp๊
Expand1x1Fire7/Conv2DConv2DSqueezeFire7/Relu:activations:0,Expand1x1Fire7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Expand1x1Fire7/Conv2Dบ
%Expand1x1Fire7/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire7_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02'
%Expand1x1Fire7/BiasAdd/ReadVariableOpล
Expand1x1Fire7/BiasAddBiasAddExpand1x1Fire7/Conv2D:output:0-Expand1x1Fire7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2
Expand1x1Fire7/BiasAdd
Expand1x1Fire7/ReluReluExpand1x1Fire7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Expand1x1Fire7/Reluร
$Expand3x3Fire7/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02&
$Expand3x3Fire7/Conv2D/ReadVariableOp๊
Expand3x3Fire7/Conv2DConv2DSqueezeFire7/Relu:activations:0,Expand3x3Fire7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Expand3x3Fire7/Conv2Dบ
%Expand3x3Fire7/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire7_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02'
%Expand3x3Fire7/BiasAdd/ReadVariableOpล
Expand3x3Fire7/BiasAddBiasAddExpand3x3Fire7/Conv2D:output:0-Expand3x3Fire7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2
Expand3x3Fire7/BiasAdd
Expand3x3Fire7/ReluReluExpand3x3Fire7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Expand3x3Fire7/Reluv
Concatenate7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate7/concat/axisใ
Concatenate7/concatConcatV2!Expand1x1Fire7/Relu:activations:0!Expand3x3Fire7/Relu:activations:0!Concatenate7/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate7/concatฝ
"SqueezeFire8/Conv2D/ReadVariableOpReadVariableOp+squeezefire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"SqueezeFire8/Conv2D/ReadVariableOpเ
SqueezeFire8/Conv2DConv2DConcatenate7/concat:output:0*SqueezeFire8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
SqueezeFire8/Conv2Dณ
#SqueezeFire8/BiasAdd/ReadVariableOpReadVariableOp,squeezefire8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#SqueezeFire8/BiasAdd/ReadVariableOpผ
SqueezeFire8/BiasAddBiasAddSqueezeFire8/Conv2D:output:0+SqueezeFire8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
SqueezeFire8/BiasAdd
SqueezeFire8/ReluReluSqueezeFire8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
SqueezeFire8/Reluร
$Expand1x1Fire8/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02&
$Expand1x1Fire8/Conv2D/ReadVariableOp๊
Expand1x1Fire8/Conv2DConv2DSqueezeFire8/Relu:activations:0,Expand1x1Fire8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand1x1Fire8/Conv2Dบ
%Expand1x1Fire8/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand1x1Fire8/BiasAdd/ReadVariableOpล
Expand1x1Fire8/BiasAddBiasAddExpand1x1Fire8/Conv2D:output:0-Expand1x1Fire8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire8/BiasAdd
Expand1x1Fire8/ReluReluExpand1x1Fire8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire8/Reluร
$Expand3x3Fire8/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02&
$Expand3x3Fire8/Conv2D/ReadVariableOp๊
Expand3x3Fire8/Conv2DConv2DSqueezeFire8/Relu:activations:0,Expand3x3Fire8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand3x3Fire8/Conv2Dบ
%Expand3x3Fire8/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand3x3Fire8/BiasAdd/ReadVariableOpล
Expand3x3Fire8/BiasAddBiasAddExpand3x3Fire8/Conv2D:output:0-Expand3x3Fire8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire8/BiasAdd
Expand3x3Fire8/ReluReluExpand3x3Fire8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire8/Reluv
Concatenate8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate8/concat/axisใ
Concatenate8/concatConcatV2!Expand1x1Fire8/Relu:activations:0!Expand3x3Fire8/Relu:activations:0!Concatenate8/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate8/concatป
MaxPool8/MaxPoolMaxPoolConcatenate8/concat:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
MaxPool8/MaxPoolฝ
"SqueezeFire9/Conv2D/ReadVariableOpReadVariableOp+squeezefire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"SqueezeFire9/Conv2D/ReadVariableOp?
SqueezeFire9/Conv2DConv2DMaxPool8/MaxPool:output:0*SqueezeFire9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
SqueezeFire9/Conv2Dณ
#SqueezeFire9/BiasAdd/ReadVariableOpReadVariableOp,squeezefire9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#SqueezeFire9/BiasAdd/ReadVariableOpผ
SqueezeFire9/BiasAddBiasAddSqueezeFire9/Conv2D:output:0+SqueezeFire9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
SqueezeFire9/BiasAdd
SqueezeFire9/ReluReluSqueezeFire9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
SqueezeFire9/Reluร
$Expand1x1Fire9/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02&
$Expand1x1Fire9/Conv2D/ReadVariableOp๊
Expand1x1Fire9/Conv2DConv2DSqueezeFire9/Relu:activations:0,Expand1x1Fire9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand1x1Fire9/Conv2Dบ
%Expand1x1Fire9/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand1x1Fire9/BiasAdd/ReadVariableOpล
Expand1x1Fire9/BiasAddBiasAddExpand1x1Fire9/Conv2D:output:0-Expand1x1Fire9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire9/BiasAdd
Expand1x1Fire9/ReluReluExpand1x1Fire9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire9/Reluร
$Expand3x3Fire9/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02&
$Expand3x3Fire9/Conv2D/ReadVariableOp๊
Expand3x3Fire9/Conv2DConv2DSqueezeFire9/Relu:activations:0,Expand3x3Fire9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand3x3Fire9/Conv2Dบ
%Expand3x3Fire9/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand3x3Fire9/BiasAdd/ReadVariableOpล
Expand3x3Fire9/BiasAddBiasAddExpand3x3Fire9/Conv2D:output:0-Expand3x3Fire9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire9/BiasAdd
Expand3x3Fire9/ReluReluExpand3x3Fire9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire9/Reluv
Concatenate9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate9/concat/axisใ
Concatenate9/concatConcatV2!Expand1x1Fire9/Relu:activations:0!Expand3x3Fire9/Relu:activations:0!Concatenate9/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate9/concatu
Dropout9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Dropout9/dropout/Constญ
Dropout9/dropout/MulMulConcatenate9/concat:output:0Dropout9/dropout/Const:output:0*
T0*0
_output_shapes
:?????????2
Dropout9/dropout/Mul|
Dropout9/dropout/ShapeShapeConcatenate9/concat:output:0*
T0*
_output_shapes
:2
Dropout9/dropout/Shapeุ
-Dropout9/dropout/random_uniform/RandomUniformRandomUniformDropout9/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????*
dtype02/
-Dropout9/dropout/random_uniform/RandomUniform
Dropout9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
Dropout9/dropout/GreaterEqual/y๋
Dropout9/dropout/GreaterEqualGreaterEqual6Dropout9/dropout/random_uniform/RandomUniform:output:0(Dropout9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????2
Dropout9/dropout/GreaterEqualฃ
Dropout9/dropout/CastCast!Dropout9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????2
Dropout9/dropout/Castง
Dropout9/dropout/Mul_1MulDropout9/dropout/Mul:z:0Dropout9/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????2
Dropout9/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? R 2
flatten_1/Const
flatten_1/ReshapeReshapeDropout9/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*)
_output_shapes
:?????????ค2
flatten_1/Reshapeฐ
 DenseFinal/MatMul/ReadVariableOpReadVariableOp)densefinal_matmul_readvariableop_resource* 
_output_shapes
:
ค*
dtype02"
 DenseFinal/MatMul/ReadVariableOpจ
DenseFinal/MatMulMatMulflatten_1/Reshape:output:0(DenseFinal/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DenseFinal/MatMulญ
!DenseFinal/BiasAdd/ReadVariableOpReadVariableOp*densefinal_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!DenseFinal/BiasAdd/ReadVariableOpญ
DenseFinal/BiasAddBiasAddDenseFinal/MatMul:product:0)DenseFinal/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DenseFinal/BiasAdd
DenseFinal/SoftmaxSoftmaxDenseFinal/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
DenseFinal/Softmaxฤ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulฬ
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp+squeezefire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpฉ
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_1/Square
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constข
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_1/mul/xค
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulฮ
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp-expand1x1fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpฉ
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_2/Square
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constข
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_2/mul/xค
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulฮ
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOp-expand3x3fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_3/Square/ReadVariableOpฉ
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_3/Square
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_3/Constข
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/Sum}
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_3/mul/xค
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/mulอ
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOp+squeezefire3_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02,
*kernel/Regularizer_4/Square/ReadVariableOpช
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer_4/Square
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_4/Constข
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/Sum}
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_4/mul/xค
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/mulฮ
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOp-expand1x1fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_5/Square/ReadVariableOpฉ
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_5/Square
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_5/Constข
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/Sum}
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_5/mul/xค
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/mulฮ
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOp-expand3x3fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_6/Square/ReadVariableOpฉ
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_6/Square
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_6/Constข
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/Sum}
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_6/mul/xค
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/mulอ
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOp+squeezefire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_7/Square/ReadVariableOpช
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_7/Square
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_7/Constข
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/Sum}
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_7/mul/xค
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/mulฯ
*kernel/Regularizer_8/Square/ReadVariableOpReadVariableOp-expand1x1fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_8/Square/ReadVariableOpช
kernel/Regularizer_8/SquareSquare2kernel/Regularizer_8/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_8/Square
kernel/Regularizer_8/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_8/Constข
kernel/Regularizer_8/SumSumkernel/Regularizer_8/Square:y:0#kernel/Regularizer_8/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/Sum}
kernel/Regularizer_8/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_8/mul/xค
kernel/Regularizer_8/mulMul#kernel/Regularizer_8/mul/x:output:0!kernel/Regularizer_8/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/mulฯ
*kernel/Regularizer_9/Square/ReadVariableOpReadVariableOp-expand3x3fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_9/Square/ReadVariableOpช
kernel/Regularizer_9/SquareSquare2kernel/Regularizer_9/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_9/Square
kernel/Regularizer_9/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_9/Constข
kernel/Regularizer_9/SumSumkernel/Regularizer_9/Square:y:0#kernel/Regularizer_9/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/Sum}
kernel/Regularizer_9/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_9/mul/xค
kernel/Regularizer_9/mulMul#kernel/Regularizer_9/mul/x:output:0!kernel/Regularizer_9/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/mulฯ
+kernel/Regularizer_10/Square/ReadVariableOpReadVariableOp+squeezefire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_10/Square/ReadVariableOpญ
kernel/Regularizer_10/SquareSquare3kernel/Regularizer_10/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_10/Square
kernel/Regularizer_10/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_10/Constฆ
kernel/Regularizer_10/SumSum kernel/Regularizer_10/Square:y:0$kernel/Regularizer_10/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/Sum
kernel/Regularizer_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_10/mul/xจ
kernel/Regularizer_10/mulMul$kernel/Regularizer_10/mul/x:output:0"kernel/Regularizer_10/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/mulั
+kernel/Regularizer_11/Square/ReadVariableOpReadVariableOp-expand1x1fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_11/Square/ReadVariableOpญ
kernel/Regularizer_11/SquareSquare3kernel/Regularizer_11/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_11/Square
kernel/Regularizer_11/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_11/Constฆ
kernel/Regularizer_11/SumSum kernel/Regularizer_11/Square:y:0$kernel/Regularizer_11/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/Sum
kernel/Regularizer_11/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_11/mul/xจ
kernel/Regularizer_11/mulMul$kernel/Regularizer_11/mul/x:output:0"kernel/Regularizer_11/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/mulั
+kernel/Regularizer_12/Square/ReadVariableOpReadVariableOp-expand3x3fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_12/Square/ReadVariableOpญ
kernel/Regularizer_12/SquareSquare3kernel/Regularizer_12/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_12/Square
kernel/Regularizer_12/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_12/Constฆ
kernel/Regularizer_12/SumSum kernel/Regularizer_12/Square:y:0$kernel/Regularizer_12/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/Sum
kernel/Regularizer_12/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_12/mul/xจ
kernel/Regularizer_12/mulMul$kernel/Regularizer_12/mul/x:output:0"kernel/Regularizer_12/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/mulฯ
+kernel/Regularizer_13/Square/ReadVariableOpReadVariableOp+squeezefire6_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_13/Square/ReadVariableOpญ
kernel/Regularizer_13/SquareSquare3kernel/Regularizer_13/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_13/Square
kernel/Regularizer_13/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_13/Constฆ
kernel/Regularizer_13/SumSum kernel/Regularizer_13/Square:y:0$kernel/Regularizer_13/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/Sum
kernel/Regularizer_13/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_13/mul/xจ
kernel/Regularizer_13/mulMul$kernel/Regularizer_13/mul/x:output:0"kernel/Regularizer_13/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/mulั
+kernel/Regularizer_14/Square/ReadVariableOpReadVariableOp-expand1x1fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_14/Square/ReadVariableOpญ
kernel/Regularizer_14/SquareSquare3kernel/Regularizer_14/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_14/Square
kernel/Regularizer_14/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_14/Constฆ
kernel/Regularizer_14/SumSum kernel/Regularizer_14/Square:y:0$kernel/Regularizer_14/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/Sum
kernel/Regularizer_14/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_14/mul/xจ
kernel/Regularizer_14/mulMul$kernel/Regularizer_14/mul/x:output:0"kernel/Regularizer_14/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/mulั
+kernel/Regularizer_15/Square/ReadVariableOpReadVariableOp-expand3x3fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_15/Square/ReadVariableOpญ
kernel/Regularizer_15/SquareSquare3kernel/Regularizer_15/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_15/Square
kernel/Regularizer_15/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_15/Constฆ
kernel/Regularizer_15/SumSum kernel/Regularizer_15/Square:y:0$kernel/Regularizer_15/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/Sum
kernel/Regularizer_15/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_15/mul/xจ
kernel/Regularizer_15/mulMul$kernel/Regularizer_15/mul/x:output:0"kernel/Regularizer_15/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/mulฯ
+kernel/Regularizer_16/Square/ReadVariableOpReadVariableOp+squeezefire7_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_16/Square/ReadVariableOpญ
kernel/Regularizer_16/SquareSquare3kernel/Regularizer_16/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_16/Square
kernel/Regularizer_16/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_16/Constฆ
kernel/Regularizer_16/SumSum kernel/Regularizer_16/Square:y:0$kernel/Regularizer_16/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/Sum
kernel/Regularizer_16/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_16/mul/xจ
kernel/Regularizer_16/mulMul$kernel/Regularizer_16/mul/x:output:0"kernel/Regularizer_16/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/mulั
+kernel/Regularizer_17/Square/ReadVariableOpReadVariableOp-expand1x1fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_17/Square/ReadVariableOpญ
kernel/Regularizer_17/SquareSquare3kernel/Regularizer_17/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_17/Square
kernel/Regularizer_17/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_17/Constฆ
kernel/Regularizer_17/SumSum kernel/Regularizer_17/Square:y:0$kernel/Regularizer_17/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/Sum
kernel/Regularizer_17/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_17/mul/xจ
kernel/Regularizer_17/mulMul$kernel/Regularizer_17/mul/x:output:0"kernel/Regularizer_17/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/mulั
+kernel/Regularizer_18/Square/ReadVariableOpReadVariableOp-expand3x3fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_18/Square/ReadVariableOpญ
kernel/Regularizer_18/SquareSquare3kernel/Regularizer_18/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_18/Square
kernel/Regularizer_18/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_18/Constฆ
kernel/Regularizer_18/SumSum kernel/Regularizer_18/Square:y:0$kernel/Regularizer_18/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/Sum
kernel/Regularizer_18/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_18/mul/xจ
kernel/Regularizer_18/mulMul$kernel/Regularizer_18/mul/x:output:0"kernel/Regularizer_18/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/mulฯ
+kernel/Regularizer_19/Square/ReadVariableOpReadVariableOp+squeezefire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_19/Square/ReadVariableOpญ
kernel/Regularizer_19/SquareSquare3kernel/Regularizer_19/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_19/Square
kernel/Regularizer_19/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_19/Constฆ
kernel/Regularizer_19/SumSum kernel/Regularizer_19/Square:y:0$kernel/Regularizer_19/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/Sum
kernel/Regularizer_19/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_19/mul/xจ
kernel/Regularizer_19/mulMul$kernel/Regularizer_19/mul/x:output:0"kernel/Regularizer_19/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/mulั
+kernel/Regularizer_20/Square/ReadVariableOpReadVariableOp-expand1x1fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_20/Square/ReadVariableOpญ
kernel/Regularizer_20/SquareSquare3kernel/Regularizer_20/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_20/Square
kernel/Regularizer_20/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_20/Constฆ
kernel/Regularizer_20/SumSum kernel/Regularizer_20/Square:y:0$kernel/Regularizer_20/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/Sum
kernel/Regularizer_20/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_20/mul/xจ
kernel/Regularizer_20/mulMul$kernel/Regularizer_20/mul/x:output:0"kernel/Regularizer_20/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/mulั
+kernel/Regularizer_21/Square/ReadVariableOpReadVariableOp-expand3x3fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_21/Square/ReadVariableOpญ
kernel/Regularizer_21/SquareSquare3kernel/Regularizer_21/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_21/Square
kernel/Regularizer_21/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_21/Constฆ
kernel/Regularizer_21/SumSum kernel/Regularizer_21/Square:y:0$kernel/Regularizer_21/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/Sum
kernel/Regularizer_21/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_21/mul/xจ
kernel/Regularizer_21/mulMul$kernel/Regularizer_21/mul/x:output:0"kernel/Regularizer_21/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/mulฯ
+kernel/Regularizer_22/Square/ReadVariableOpReadVariableOp+squeezefire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_22/Square/ReadVariableOpญ
kernel/Regularizer_22/SquareSquare3kernel/Regularizer_22/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_22/Square
kernel/Regularizer_22/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_22/Constฆ
kernel/Regularizer_22/SumSum kernel/Regularizer_22/Square:y:0$kernel/Regularizer_22/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/Sum
kernel/Regularizer_22/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_22/mul/xจ
kernel/Regularizer_22/mulMul$kernel/Regularizer_22/mul/x:output:0"kernel/Regularizer_22/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/mulั
+kernel/Regularizer_23/Square/ReadVariableOpReadVariableOp-expand1x1fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_23/Square/ReadVariableOpญ
kernel/Regularizer_23/SquareSquare3kernel/Regularizer_23/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_23/Square
kernel/Regularizer_23/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_23/Constฆ
kernel/Regularizer_23/SumSum kernel/Regularizer_23/Square:y:0$kernel/Regularizer_23/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/Sum
kernel/Regularizer_23/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_23/mul/xจ
kernel/Regularizer_23/mulMul$kernel/Regularizer_23/mul/x:output:0"kernel/Regularizer_23/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/mulั
+kernel/Regularizer_24/Square/ReadVariableOpReadVariableOp-expand3x3fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_24/Square/ReadVariableOpญ
kernel/Regularizer_24/SquareSquare3kernel/Regularizer_24/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_24/Square
kernel/Regularizer_24/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_24/Constฆ
kernel/Regularizer_24/SumSum kernel/Regularizer_24/Square:y:0$kernel/Regularizer_24/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/Sum
kernel/Regularizer_24/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_24/mul/xจ
kernel/Regularizer_24/mulMul$kernel/Regularizer_24/mul/x:output:0"kernel/Regularizer_24/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/mulp
IdentityIdentityDenseFinal/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ:::::::::::::::::::::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
๋
r
H__inference_Concatenate9_layer_call_and_return_conditional_losses_265929

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs
๓
t
H__inference_Concatenate5_layer_call_and_return_conditional_losses_268940
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:Z V
0
_output_shapes
:?????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ฟ	
f
__inference_loss_fn_8_2695395
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:


-__inference_SqueezeFire7_layer_call_fn_269087

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_2656092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ฝ	
f
__inference_loss_fn_1_2694625
1kernel_regularizer_square_readvariableop_resource
identityฮ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?๚
ว
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_267163

inputs
conv2d_1_266869
conv2d_1_266871
squeezefire2_266875
squeezefire2_266877
expand1x1fire2_266880
expand1x1fire2_266882
expand3x3fire2_266885
expand3x3fire2_266887
squeezefire3_266891
squeezefire3_266893
expand1x1fire3_266896
expand1x1fire3_266898
expand3x3fire3_266901
expand3x3fire3_266903
squeezefire4_266907
squeezefire4_266909
expand1x1fire4_266912
expand1x1fire4_266914
expand3x3fire4_266917
expand3x3fire4_266919
squeezefire5_266924
squeezefire5_266926
expand1x1fire5_266929
expand1x1fire5_266931
expand3x3fire5_266934
expand3x3fire5_266936
squeezefire6_266940
squeezefire6_266942
expand1x1fire6_266945
expand1x1fire6_266947
expand3x3fire6_266950
expand3x3fire6_266952
squeezefire7_266956
squeezefire7_266958
expand1x1fire7_266961
expand1x1fire7_266963
expand3x3fire7_266966
expand3x3fire7_266968
squeezefire8_266972
squeezefire8_266974
expand1x1fire8_266977
expand1x1fire8_266979
expand3x3fire8_266982
expand3x3fire8_266984
squeezefire9_266989
squeezefire9_266991
expand1x1fire9_266994
expand1x1fire9_266996
expand3x3fire9_266999
expand3x3fire9_267001
densefinal_267007
densefinal_267009
identityข Conv2D_1/StatefulPartitionedCallข"DenseFinal/StatefulPartitionedCallข&Expand1x1Fire2/StatefulPartitionedCallข&Expand1x1Fire3/StatefulPartitionedCallข&Expand1x1Fire4/StatefulPartitionedCallข&Expand1x1Fire5/StatefulPartitionedCallข&Expand1x1Fire6/StatefulPartitionedCallข&Expand1x1Fire7/StatefulPartitionedCallข&Expand1x1Fire8/StatefulPartitionedCallข&Expand1x1Fire9/StatefulPartitionedCallข&Expand3x3Fire2/StatefulPartitionedCallข&Expand3x3Fire3/StatefulPartitionedCallข&Expand3x3Fire4/StatefulPartitionedCallข&Expand3x3Fire5/StatefulPartitionedCallข&Expand3x3Fire6/StatefulPartitionedCallข&Expand3x3Fire7/StatefulPartitionedCallข&Expand3x3Fire8/StatefulPartitionedCallข&Expand3x3Fire9/StatefulPartitionedCallข$SqueezeFire2/StatefulPartitionedCallข$SqueezeFire3/StatefulPartitionedCallข$SqueezeFire4/StatefulPartitionedCallข$SqueezeFire5/StatefulPartitionedCallข$SqueezeFire6/StatefulPartitionedCallข$SqueezeFire7/StatefulPartitionedCallข$SqueezeFire8/StatefulPartitionedCallข$SqueezeFire9/StatefulPartitionedCall
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_266869conv2d_1_266871*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_2649992"
 Conv2D_1/StatefulPartitionedCall
MaxPool1/PartitionedCallPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool1_layer_call_and_return_conditional_losses_2649482
MaxPool1/PartitionedCallฮ
$SqueezeFire2/StatefulPartitionedCallStatefulPartitionedCall!MaxPool1/PartitionedCall:output:0squeezefire2_266875squeezefire2_266877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_2650332&
$SqueezeFire2/StatefulPartitionedCallไ
&Expand1x1Fire2/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire2/StatefulPartitionedCall:output:0expand1x1fire2_266880expand1x1fire2_266882*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_2650662(
&Expand1x1Fire2/StatefulPartitionedCallไ
&Expand3x3Fire2/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire2/StatefulPartitionedCall:output:0expand3x3fire2_266885expand3x3fire2_266887*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_2650992(
&Expand3x3Fire2/StatefulPartitionedCallว
Concatenate2/PartitionedCallPartitionedCall/Expand1x1Fire2/StatefulPartitionedCall:output:0/Expand3x3Fire2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate2_layer_call_and_return_conditional_losses_2651222
Concatenate2/PartitionedCallา
$SqueezeFire3/StatefulPartitionedCallStatefulPartitionedCall%Concatenate2/PartitionedCall:output:0squeezefire3_266891squeezefire3_266893*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_2651482&
$SqueezeFire3/StatefulPartitionedCallไ
&Expand1x1Fire3/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire3/StatefulPartitionedCall:output:0expand1x1fire3_266896expand1x1fire3_266898*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_2651812(
&Expand1x1Fire3/StatefulPartitionedCallไ
&Expand3x3Fire3/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire3/StatefulPartitionedCall:output:0expand3x3fire3_266901expand3x3fire3_266903*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_2652142(
&Expand3x3Fire3/StatefulPartitionedCallว
Concatenate3/PartitionedCallPartitionedCall/Expand1x1Fire3/StatefulPartitionedCall:output:0/Expand3x3Fire3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate3_layer_call_and_return_conditional_losses_2652372
Concatenate3/PartitionedCallา
$SqueezeFire4/StatefulPartitionedCallStatefulPartitionedCall%Concatenate3/PartitionedCall:output:0squeezefire4_266907squeezefire4_266909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_2652632&
$SqueezeFire4/StatefulPartitionedCallๅ
&Expand1x1Fire4/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire4/StatefulPartitionedCall:output:0expand1x1fire4_266912expand1x1fire4_266914*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_2652962(
&Expand1x1Fire4/StatefulPartitionedCallๅ
&Expand3x3Fire4/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire4/StatefulPartitionedCall:output:0expand3x3fire4_266917expand3x3fire4_266919*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_2653292(
&Expand3x3Fire4/StatefulPartitionedCallว
Concatenate4/PartitionedCallPartitionedCall/Expand1x1Fire4/StatefulPartitionedCall:output:0/Expand3x3Fire4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate4_layer_call_and_return_conditional_losses_2653522
Concatenate4/PartitionedCall?
MaxPool4/PartitionedCallPartitionedCall%Concatenate4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool4_layer_call_and_return_conditional_losses_2649602
MaxPool4/PartitionedCallฮ
$SqueezeFire5/StatefulPartitionedCallStatefulPartitionedCall!MaxPool4/PartitionedCall:output:0squeezefire5_266924squeezefire5_266926*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_2653792&
$SqueezeFire5/StatefulPartitionedCallๅ
&Expand1x1Fire5/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire5/StatefulPartitionedCall:output:0expand1x1fire5_266929expand1x1fire5_266931*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_2654122(
&Expand1x1Fire5/StatefulPartitionedCallๅ
&Expand3x3Fire5/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire5/StatefulPartitionedCall:output:0expand3x3fire5_266934expand3x3fire5_266936*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_2654452(
&Expand3x3Fire5/StatefulPartitionedCallว
Concatenate5/PartitionedCallPartitionedCall/Expand1x1Fire5/StatefulPartitionedCall:output:0/Expand3x3Fire5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate5_layer_call_and_return_conditional_losses_2654682
Concatenate5/PartitionedCallา
$SqueezeFire6/StatefulPartitionedCallStatefulPartitionedCall%Concatenate5/PartitionedCall:output:0squeezefire6_266940squeezefire6_266942*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_2654942&
$SqueezeFire6/StatefulPartitionedCallๅ
&Expand1x1Fire6/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire6/StatefulPartitionedCall:output:0expand1x1fire6_266945expand1x1fire6_266947*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_2655272(
&Expand1x1Fire6/StatefulPartitionedCallๅ
&Expand3x3Fire6/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire6/StatefulPartitionedCall:output:0expand3x3fire6_266950expand3x3fire6_266952*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_2655602(
&Expand3x3Fire6/StatefulPartitionedCallว
Concatenate6/PartitionedCallPartitionedCall/Expand1x1Fire6/StatefulPartitionedCall:output:0/Expand3x3Fire6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate6_layer_call_and_return_conditional_losses_2655832
Concatenate6/PartitionedCallา
$SqueezeFire7/StatefulPartitionedCallStatefulPartitionedCall%Concatenate6/PartitionedCall:output:0squeezefire7_266956squeezefire7_266958*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_2656092&
$SqueezeFire7/StatefulPartitionedCallๅ
&Expand1x1Fire7/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire7/StatefulPartitionedCall:output:0expand1x1fire7_266961expand1x1fire7_266963*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_2656422(
&Expand1x1Fire7/StatefulPartitionedCallๅ
&Expand3x3Fire7/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire7/StatefulPartitionedCall:output:0expand3x3fire7_266966expand3x3fire7_266968*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_2656752(
&Expand3x3Fire7/StatefulPartitionedCallว
Concatenate7/PartitionedCallPartitionedCall/Expand1x1Fire7/StatefulPartitionedCall:output:0/Expand3x3Fire7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate7_layer_call_and_return_conditional_losses_2656982
Concatenate7/PartitionedCallา
$SqueezeFire8/StatefulPartitionedCallStatefulPartitionedCall%Concatenate7/PartitionedCall:output:0squeezefire8_266972squeezefire8_266974*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_2657242&
$SqueezeFire8/StatefulPartitionedCallๅ
&Expand1x1Fire8/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire8/StatefulPartitionedCall:output:0expand1x1fire8_266977expand1x1fire8_266979*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_2657572(
&Expand1x1Fire8/StatefulPartitionedCallๅ
&Expand3x3Fire8/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire8/StatefulPartitionedCall:output:0expand3x3fire8_266982expand3x3fire8_266984*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_2657902(
&Expand3x3Fire8/StatefulPartitionedCallว
Concatenate8/PartitionedCallPartitionedCall/Expand1x1Fire8/StatefulPartitionedCall:output:0/Expand3x3Fire8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate8_layer_call_and_return_conditional_losses_2658132
Concatenate8/PartitionedCall?
MaxPool8/PartitionedCallPartitionedCall%Concatenate8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool8_layer_call_and_return_conditional_losses_2649722
MaxPool8/PartitionedCallฮ
$SqueezeFire9/StatefulPartitionedCallStatefulPartitionedCall!MaxPool8/PartitionedCall:output:0squeezefire9_266989squeezefire9_266991*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_2658402&
$SqueezeFire9/StatefulPartitionedCallๅ
&Expand1x1Fire9/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire9/StatefulPartitionedCall:output:0expand1x1fire9_266994expand1x1fire9_266996*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_2658732(
&Expand1x1Fire9/StatefulPartitionedCallๅ
&Expand3x3Fire9/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire9/StatefulPartitionedCall:output:0expand3x3fire9_266999expand3x3fire9_267001*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_2659062(
&Expand3x3Fire9/StatefulPartitionedCallว
Concatenate9/PartitionedCallPartitionedCall/Expand1x1Fire9/StatefulPartitionedCall:output:0/Expand3x3Fire9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate9_layer_call_and_return_conditional_losses_2659292
Concatenate9/PartitionedCall?
Dropout9/PartitionedCallPartitionedCall%Concatenate9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dropout9_layer_call_and_return_conditional_losses_2659552
Dropout9/PartitionedCall๗
flatten_1/PartitionedCallPartitionedCall!Dropout9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2659742
flatten_1/PartitionedCallฝ
"DenseFinal/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0densefinal_267007densefinal_267009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_DenseFinal_layer_call_and_return_conditional_losses_2659932$
"DenseFinal/StatefulPartitionedCallฌ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_266869*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulด
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOpsqueezefire2_266875*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpฉ
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_1/Square
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constข
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_1/mul/xค
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulถ
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOpexpand1x1fire2_266880*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpฉ
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_2/Square
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constข
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_2/mul/xค
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulถ
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOpexpand3x3fire2_266885*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_3/Square/ReadVariableOpฉ
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_3/Square
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_3/Constข
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/Sum}
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_3/mul/xค
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/mulต
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOpsqueezefire3_266891*'
_output_shapes
:*
dtype02,
*kernel/Regularizer_4/Square/ReadVariableOpช
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer_4/Square
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_4/Constข
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/Sum}
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_4/mul/xค
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/mulถ
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOpexpand1x1fire3_266896*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_5/Square/ReadVariableOpฉ
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_5/Square
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_5/Constข
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/Sum}
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_5/mul/xค
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/mulถ
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOpexpand3x3fire3_266901*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_6/Square/ReadVariableOpฉ
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_6/Square
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_6/Constข
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/Sum}
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_6/mul/xค
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/mulต
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOpsqueezefire4_266907*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_7/Square/ReadVariableOpช
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_7/Square
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_7/Constข
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/Sum}
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_7/mul/xค
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/mulท
*kernel/Regularizer_8/Square/ReadVariableOpReadVariableOpexpand1x1fire4_266912*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_8/Square/ReadVariableOpช
kernel/Regularizer_8/SquareSquare2kernel/Regularizer_8/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_8/Square
kernel/Regularizer_8/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_8/Constข
kernel/Regularizer_8/SumSumkernel/Regularizer_8/Square:y:0#kernel/Regularizer_8/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/Sum}
kernel/Regularizer_8/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_8/mul/xค
kernel/Regularizer_8/mulMul#kernel/Regularizer_8/mul/x:output:0!kernel/Regularizer_8/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/mulท
*kernel/Regularizer_9/Square/ReadVariableOpReadVariableOpexpand3x3fire4_266917*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_9/Square/ReadVariableOpช
kernel/Regularizer_9/SquareSquare2kernel/Regularizer_9/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_9/Square
kernel/Regularizer_9/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_9/Constข
kernel/Regularizer_9/SumSumkernel/Regularizer_9/Square:y:0#kernel/Regularizer_9/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/Sum}
kernel/Regularizer_9/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_9/mul/xค
kernel/Regularizer_9/mulMul#kernel/Regularizer_9/mul/x:output:0!kernel/Regularizer_9/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/mulท
+kernel/Regularizer_10/Square/ReadVariableOpReadVariableOpsqueezefire5_266924*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_10/Square/ReadVariableOpญ
kernel/Regularizer_10/SquareSquare3kernel/Regularizer_10/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_10/Square
kernel/Regularizer_10/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_10/Constฆ
kernel/Regularizer_10/SumSum kernel/Regularizer_10/Square:y:0$kernel/Regularizer_10/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/Sum
kernel/Regularizer_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_10/mul/xจ
kernel/Regularizer_10/mulMul$kernel/Regularizer_10/mul/x:output:0"kernel/Regularizer_10/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/mulน
+kernel/Regularizer_11/Square/ReadVariableOpReadVariableOpexpand1x1fire5_266929*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_11/Square/ReadVariableOpญ
kernel/Regularizer_11/SquareSquare3kernel/Regularizer_11/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_11/Square
kernel/Regularizer_11/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_11/Constฆ
kernel/Regularizer_11/SumSum kernel/Regularizer_11/Square:y:0$kernel/Regularizer_11/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/Sum
kernel/Regularizer_11/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_11/mul/xจ
kernel/Regularizer_11/mulMul$kernel/Regularizer_11/mul/x:output:0"kernel/Regularizer_11/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/mulน
+kernel/Regularizer_12/Square/ReadVariableOpReadVariableOpexpand3x3fire5_266934*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_12/Square/ReadVariableOpญ
kernel/Regularizer_12/SquareSquare3kernel/Regularizer_12/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_12/Square
kernel/Regularizer_12/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_12/Constฆ
kernel/Regularizer_12/SumSum kernel/Regularizer_12/Square:y:0$kernel/Regularizer_12/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/Sum
kernel/Regularizer_12/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_12/mul/xจ
kernel/Regularizer_12/mulMul$kernel/Regularizer_12/mul/x:output:0"kernel/Regularizer_12/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/mulท
+kernel/Regularizer_13/Square/ReadVariableOpReadVariableOpsqueezefire6_266940*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_13/Square/ReadVariableOpญ
kernel/Regularizer_13/SquareSquare3kernel/Regularizer_13/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_13/Square
kernel/Regularizer_13/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_13/Constฆ
kernel/Regularizer_13/SumSum kernel/Regularizer_13/Square:y:0$kernel/Regularizer_13/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/Sum
kernel/Regularizer_13/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_13/mul/xจ
kernel/Regularizer_13/mulMul$kernel/Regularizer_13/mul/x:output:0"kernel/Regularizer_13/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/mulน
+kernel/Regularizer_14/Square/ReadVariableOpReadVariableOpexpand1x1fire6_266945*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_14/Square/ReadVariableOpญ
kernel/Regularizer_14/SquareSquare3kernel/Regularizer_14/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_14/Square
kernel/Regularizer_14/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_14/Constฆ
kernel/Regularizer_14/SumSum kernel/Regularizer_14/Square:y:0$kernel/Regularizer_14/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/Sum
kernel/Regularizer_14/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_14/mul/xจ
kernel/Regularizer_14/mulMul$kernel/Regularizer_14/mul/x:output:0"kernel/Regularizer_14/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/mulน
+kernel/Regularizer_15/Square/ReadVariableOpReadVariableOpexpand3x3fire6_266950*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_15/Square/ReadVariableOpญ
kernel/Regularizer_15/SquareSquare3kernel/Regularizer_15/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_15/Square
kernel/Regularizer_15/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_15/Constฆ
kernel/Regularizer_15/SumSum kernel/Regularizer_15/Square:y:0$kernel/Regularizer_15/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/Sum
kernel/Regularizer_15/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_15/mul/xจ
kernel/Regularizer_15/mulMul$kernel/Regularizer_15/mul/x:output:0"kernel/Regularizer_15/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/mulท
+kernel/Regularizer_16/Square/ReadVariableOpReadVariableOpsqueezefire7_266956*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_16/Square/ReadVariableOpญ
kernel/Regularizer_16/SquareSquare3kernel/Regularizer_16/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_16/Square
kernel/Regularizer_16/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_16/Constฆ
kernel/Regularizer_16/SumSum kernel/Regularizer_16/Square:y:0$kernel/Regularizer_16/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/Sum
kernel/Regularizer_16/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_16/mul/xจ
kernel/Regularizer_16/mulMul$kernel/Regularizer_16/mul/x:output:0"kernel/Regularizer_16/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/mulน
+kernel/Regularizer_17/Square/ReadVariableOpReadVariableOpexpand1x1fire7_266961*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_17/Square/ReadVariableOpญ
kernel/Regularizer_17/SquareSquare3kernel/Regularizer_17/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_17/Square
kernel/Regularizer_17/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_17/Constฆ
kernel/Regularizer_17/SumSum kernel/Regularizer_17/Square:y:0$kernel/Regularizer_17/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/Sum
kernel/Regularizer_17/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_17/mul/xจ
kernel/Regularizer_17/mulMul$kernel/Regularizer_17/mul/x:output:0"kernel/Regularizer_17/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/mulน
+kernel/Regularizer_18/Square/ReadVariableOpReadVariableOpexpand3x3fire7_266966*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_18/Square/ReadVariableOpญ
kernel/Regularizer_18/SquareSquare3kernel/Regularizer_18/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_18/Square
kernel/Regularizer_18/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_18/Constฆ
kernel/Regularizer_18/SumSum kernel/Regularizer_18/Square:y:0$kernel/Regularizer_18/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/Sum
kernel/Regularizer_18/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_18/mul/xจ
kernel/Regularizer_18/mulMul$kernel/Regularizer_18/mul/x:output:0"kernel/Regularizer_18/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/mulท
+kernel/Regularizer_19/Square/ReadVariableOpReadVariableOpsqueezefire8_266972*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_19/Square/ReadVariableOpญ
kernel/Regularizer_19/SquareSquare3kernel/Regularizer_19/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_19/Square
kernel/Regularizer_19/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_19/Constฆ
kernel/Regularizer_19/SumSum kernel/Regularizer_19/Square:y:0$kernel/Regularizer_19/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/Sum
kernel/Regularizer_19/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_19/mul/xจ
kernel/Regularizer_19/mulMul$kernel/Regularizer_19/mul/x:output:0"kernel/Regularizer_19/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/mulน
+kernel/Regularizer_20/Square/ReadVariableOpReadVariableOpexpand1x1fire8_266977*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_20/Square/ReadVariableOpญ
kernel/Regularizer_20/SquareSquare3kernel/Regularizer_20/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_20/Square
kernel/Regularizer_20/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_20/Constฆ
kernel/Regularizer_20/SumSum kernel/Regularizer_20/Square:y:0$kernel/Regularizer_20/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/Sum
kernel/Regularizer_20/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_20/mul/xจ
kernel/Regularizer_20/mulMul$kernel/Regularizer_20/mul/x:output:0"kernel/Regularizer_20/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/mulน
+kernel/Regularizer_21/Square/ReadVariableOpReadVariableOpexpand3x3fire8_266982*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_21/Square/ReadVariableOpญ
kernel/Regularizer_21/SquareSquare3kernel/Regularizer_21/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_21/Square
kernel/Regularizer_21/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_21/Constฆ
kernel/Regularizer_21/SumSum kernel/Regularizer_21/Square:y:0$kernel/Regularizer_21/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/Sum
kernel/Regularizer_21/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_21/mul/xจ
kernel/Regularizer_21/mulMul$kernel/Regularizer_21/mul/x:output:0"kernel/Regularizer_21/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/mulท
+kernel/Regularizer_22/Square/ReadVariableOpReadVariableOpsqueezefire9_266989*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_22/Square/ReadVariableOpญ
kernel/Regularizer_22/SquareSquare3kernel/Regularizer_22/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_22/Square
kernel/Regularizer_22/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_22/Constฆ
kernel/Regularizer_22/SumSum kernel/Regularizer_22/Square:y:0$kernel/Regularizer_22/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/Sum
kernel/Regularizer_22/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_22/mul/xจ
kernel/Regularizer_22/mulMul$kernel/Regularizer_22/mul/x:output:0"kernel/Regularizer_22/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/mulน
+kernel/Regularizer_23/Square/ReadVariableOpReadVariableOpexpand1x1fire9_266994*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_23/Square/ReadVariableOpญ
kernel/Regularizer_23/SquareSquare3kernel/Regularizer_23/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_23/Square
kernel/Regularizer_23/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_23/Constฆ
kernel/Regularizer_23/SumSum kernel/Regularizer_23/Square:y:0$kernel/Regularizer_23/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/Sum
kernel/Regularizer_23/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_23/mul/xจ
kernel/Regularizer_23/mulMul$kernel/Regularizer_23/mul/x:output:0"kernel/Regularizer_23/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/mulน
+kernel/Regularizer_24/Square/ReadVariableOpReadVariableOpexpand3x3fire9_266999*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_24/Square/ReadVariableOpญ
kernel/Regularizer_24/SquareSquare3kernel/Regularizer_24/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_24/Square
kernel/Regularizer_24/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_24/Constฆ
kernel/Regularizer_24/SumSum kernel/Regularizer_24/Square:y:0$kernel/Regularizer_24/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/Sum
kernel/Regularizer_24/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_24/mul/xจ
kernel/Regularizer_24/mulMul$kernel/Regularizer_24/mul/x:output:0"kernel/Regularizer_24/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/mul	
IdentityIdentity+DenseFinal/StatefulPartitionedCall:output:0!^Conv2D_1/StatefulPartitionedCall#^DenseFinal/StatefulPartitionedCall'^Expand1x1Fire2/StatefulPartitionedCall'^Expand1x1Fire3/StatefulPartitionedCall'^Expand1x1Fire4/StatefulPartitionedCall'^Expand1x1Fire5/StatefulPartitionedCall'^Expand1x1Fire6/StatefulPartitionedCall'^Expand1x1Fire7/StatefulPartitionedCall'^Expand1x1Fire8/StatefulPartitionedCall'^Expand1x1Fire9/StatefulPartitionedCall'^Expand3x3Fire2/StatefulPartitionedCall'^Expand3x3Fire3/StatefulPartitionedCall'^Expand3x3Fire4/StatefulPartitionedCall'^Expand3x3Fire5/StatefulPartitionedCall'^Expand3x3Fire6/StatefulPartitionedCall'^Expand3x3Fire7/StatefulPartitionedCall'^Expand3x3Fire8/StatefulPartitionedCall'^Expand3x3Fire9/StatefulPartitionedCall%^SqueezeFire2/StatefulPartitionedCall%^SqueezeFire3/StatefulPartitionedCall%^SqueezeFire4/StatefulPartitionedCall%^SqueezeFire5/StatefulPartitionedCall%^SqueezeFire6/StatefulPartitionedCall%^SqueezeFire7/StatefulPartitionedCall%^SqueezeFire8/StatefulPartitionedCall%^SqueezeFire9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2H
"DenseFinal/StatefulPartitionedCall"DenseFinal/StatefulPartitionedCall2P
&Expand1x1Fire2/StatefulPartitionedCall&Expand1x1Fire2/StatefulPartitionedCall2P
&Expand1x1Fire3/StatefulPartitionedCall&Expand1x1Fire3/StatefulPartitionedCall2P
&Expand1x1Fire4/StatefulPartitionedCall&Expand1x1Fire4/StatefulPartitionedCall2P
&Expand1x1Fire5/StatefulPartitionedCall&Expand1x1Fire5/StatefulPartitionedCall2P
&Expand1x1Fire6/StatefulPartitionedCall&Expand1x1Fire6/StatefulPartitionedCall2P
&Expand1x1Fire7/StatefulPartitionedCall&Expand1x1Fire7/StatefulPartitionedCall2P
&Expand1x1Fire8/StatefulPartitionedCall&Expand1x1Fire8/StatefulPartitionedCall2P
&Expand1x1Fire9/StatefulPartitionedCall&Expand1x1Fire9/StatefulPartitionedCall2P
&Expand3x3Fire2/StatefulPartitionedCall&Expand3x3Fire2/StatefulPartitionedCall2P
&Expand3x3Fire3/StatefulPartitionedCall&Expand3x3Fire3/StatefulPartitionedCall2P
&Expand3x3Fire4/StatefulPartitionedCall&Expand3x3Fire4/StatefulPartitionedCall2P
&Expand3x3Fire5/StatefulPartitionedCall&Expand3x3Fire5/StatefulPartitionedCall2P
&Expand3x3Fire6/StatefulPartitionedCall&Expand3x3Fire6/StatefulPartitionedCall2P
&Expand3x3Fire7/StatefulPartitionedCall&Expand3x3Fire7/StatefulPartitionedCall2P
&Expand3x3Fire8/StatefulPartitionedCall&Expand3x3Fire8/StatefulPartitionedCall2P
&Expand3x3Fire9/StatefulPartitionedCall&Expand3x3Fire9/StatefulPartitionedCall2L
$SqueezeFire2/StatefulPartitionedCall$SqueezeFire2/StatefulPartitionedCall2L
$SqueezeFire3/StatefulPartitionedCall$SqueezeFire3/StatefulPartitionedCall2L
$SqueezeFire4/StatefulPartitionedCall$SqueezeFire4/StatefulPartitionedCall2L
$SqueezeFire5/StatefulPartitionedCall$SqueezeFire5/StatefulPartitionedCall2L
$SqueezeFire6/StatefulPartitionedCall$SqueezeFire6/StatefulPartitionedCall2L
$SqueezeFire7/StatefulPartitionedCall$SqueezeFire7/StatefulPartitionedCall2L
$SqueezeFire8/StatefulPartitionedCall$SqueezeFire8/StatefulPartitionedCall2L
$SqueezeFire9/StatefulPartitionedCall$SqueezeFire9/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
ิ
Y
-__inference_Concatenate3_layer_call_fn_268728
inputs_0
inputs_1
identity฿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate3_layer_call_and_return_conditional_losses_2652372
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*I
_input_shapes8
6:?????????77@:?????????77@:Y U
/
_output_shapes
:?????????77@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????77@
"
_user_specified_name
inputs/1
บ
ฒ
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_265214

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77:::W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
?
้
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_266160	
input
conv2d_1_265010
conv2d_1_265012
squeezefire2_265044
squeezefire2_265046
expand1x1fire2_265077
expand1x1fire2_265079
expand3x3fire2_265110
expand3x3fire2_265112
squeezefire3_265159
squeezefire3_265161
expand1x1fire3_265192
expand1x1fire3_265194
expand3x3fire3_265225
expand3x3fire3_265227
squeezefire4_265274
squeezefire4_265276
expand1x1fire4_265307
expand1x1fire4_265309
expand3x3fire4_265340
expand3x3fire4_265342
squeezefire5_265390
squeezefire5_265392
expand1x1fire5_265423
expand1x1fire5_265425
expand3x3fire5_265456
expand3x3fire5_265458
squeezefire6_265505
squeezefire6_265507
expand1x1fire6_265538
expand1x1fire6_265540
expand3x3fire6_265571
expand3x3fire6_265573
squeezefire7_265620
squeezefire7_265622
expand1x1fire7_265653
expand1x1fire7_265655
expand3x3fire7_265686
expand3x3fire7_265688
squeezefire8_265735
squeezefire8_265737
expand1x1fire8_265768
expand1x1fire8_265770
expand3x3fire8_265801
expand3x3fire8_265803
squeezefire9_265851
squeezefire9_265853
expand1x1fire9_265884
expand1x1fire9_265886
expand3x3fire9_265917
expand3x3fire9_265919
densefinal_266004
densefinal_266006
identityข Conv2D_1/StatefulPartitionedCallข"DenseFinal/StatefulPartitionedCallข Dropout9/StatefulPartitionedCallข&Expand1x1Fire2/StatefulPartitionedCallข&Expand1x1Fire3/StatefulPartitionedCallข&Expand1x1Fire4/StatefulPartitionedCallข&Expand1x1Fire5/StatefulPartitionedCallข&Expand1x1Fire6/StatefulPartitionedCallข&Expand1x1Fire7/StatefulPartitionedCallข&Expand1x1Fire8/StatefulPartitionedCallข&Expand1x1Fire9/StatefulPartitionedCallข&Expand3x3Fire2/StatefulPartitionedCallข&Expand3x3Fire3/StatefulPartitionedCallข&Expand3x3Fire4/StatefulPartitionedCallข&Expand3x3Fire5/StatefulPartitionedCallข&Expand3x3Fire6/StatefulPartitionedCallข&Expand3x3Fire7/StatefulPartitionedCallข&Expand3x3Fire8/StatefulPartitionedCallข&Expand3x3Fire9/StatefulPartitionedCallข$SqueezeFire2/StatefulPartitionedCallข$SqueezeFire3/StatefulPartitionedCallข$SqueezeFire4/StatefulPartitionedCallข$SqueezeFire5/StatefulPartitionedCallข$SqueezeFire6/StatefulPartitionedCallข$SqueezeFire7/StatefulPartitionedCallข$SqueezeFire8/StatefulPartitionedCallข$SqueezeFire9/StatefulPartitionedCall
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_1_265010conv2d_1_265012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_2649992"
 Conv2D_1/StatefulPartitionedCall
MaxPool1/PartitionedCallPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool1_layer_call_and_return_conditional_losses_2649482
MaxPool1/PartitionedCallฮ
$SqueezeFire2/StatefulPartitionedCallStatefulPartitionedCall!MaxPool1/PartitionedCall:output:0squeezefire2_265044squeezefire2_265046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_2650332&
$SqueezeFire2/StatefulPartitionedCallไ
&Expand1x1Fire2/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire2/StatefulPartitionedCall:output:0expand1x1fire2_265077expand1x1fire2_265079*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_2650662(
&Expand1x1Fire2/StatefulPartitionedCallไ
&Expand3x3Fire2/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire2/StatefulPartitionedCall:output:0expand3x3fire2_265110expand3x3fire2_265112*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_2650992(
&Expand3x3Fire2/StatefulPartitionedCallว
Concatenate2/PartitionedCallPartitionedCall/Expand1x1Fire2/StatefulPartitionedCall:output:0/Expand3x3Fire2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate2_layer_call_and_return_conditional_losses_2651222
Concatenate2/PartitionedCallา
$SqueezeFire3/StatefulPartitionedCallStatefulPartitionedCall%Concatenate2/PartitionedCall:output:0squeezefire3_265159squeezefire3_265161*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_2651482&
$SqueezeFire3/StatefulPartitionedCallไ
&Expand1x1Fire3/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire3/StatefulPartitionedCall:output:0expand1x1fire3_265192expand1x1fire3_265194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_2651812(
&Expand1x1Fire3/StatefulPartitionedCallไ
&Expand3x3Fire3/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire3/StatefulPartitionedCall:output:0expand3x3fire3_265225expand3x3fire3_265227*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_2652142(
&Expand3x3Fire3/StatefulPartitionedCallว
Concatenate3/PartitionedCallPartitionedCall/Expand1x1Fire3/StatefulPartitionedCall:output:0/Expand3x3Fire3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate3_layer_call_and_return_conditional_losses_2652372
Concatenate3/PartitionedCallา
$SqueezeFire4/StatefulPartitionedCallStatefulPartitionedCall%Concatenate3/PartitionedCall:output:0squeezefire4_265274squeezefire4_265276*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_2652632&
$SqueezeFire4/StatefulPartitionedCallๅ
&Expand1x1Fire4/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire4/StatefulPartitionedCall:output:0expand1x1fire4_265307expand1x1fire4_265309*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_2652962(
&Expand1x1Fire4/StatefulPartitionedCallๅ
&Expand3x3Fire4/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire4/StatefulPartitionedCall:output:0expand3x3fire4_265340expand3x3fire4_265342*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_2653292(
&Expand3x3Fire4/StatefulPartitionedCallว
Concatenate4/PartitionedCallPartitionedCall/Expand1x1Fire4/StatefulPartitionedCall:output:0/Expand3x3Fire4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate4_layer_call_and_return_conditional_losses_2653522
Concatenate4/PartitionedCall?
MaxPool4/PartitionedCallPartitionedCall%Concatenate4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool4_layer_call_and_return_conditional_losses_2649602
MaxPool4/PartitionedCallฮ
$SqueezeFire5/StatefulPartitionedCallStatefulPartitionedCall!MaxPool4/PartitionedCall:output:0squeezefire5_265390squeezefire5_265392*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_2653792&
$SqueezeFire5/StatefulPartitionedCallๅ
&Expand1x1Fire5/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire5/StatefulPartitionedCall:output:0expand1x1fire5_265423expand1x1fire5_265425*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_2654122(
&Expand1x1Fire5/StatefulPartitionedCallๅ
&Expand3x3Fire5/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire5/StatefulPartitionedCall:output:0expand3x3fire5_265456expand3x3fire5_265458*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_2654452(
&Expand3x3Fire5/StatefulPartitionedCallว
Concatenate5/PartitionedCallPartitionedCall/Expand1x1Fire5/StatefulPartitionedCall:output:0/Expand3x3Fire5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate5_layer_call_and_return_conditional_losses_2654682
Concatenate5/PartitionedCallา
$SqueezeFire6/StatefulPartitionedCallStatefulPartitionedCall%Concatenate5/PartitionedCall:output:0squeezefire6_265505squeezefire6_265507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_2654942&
$SqueezeFire6/StatefulPartitionedCallๅ
&Expand1x1Fire6/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire6/StatefulPartitionedCall:output:0expand1x1fire6_265538expand1x1fire6_265540*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_2655272(
&Expand1x1Fire6/StatefulPartitionedCallๅ
&Expand3x3Fire6/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire6/StatefulPartitionedCall:output:0expand3x3fire6_265571expand3x3fire6_265573*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_2655602(
&Expand3x3Fire6/StatefulPartitionedCallว
Concatenate6/PartitionedCallPartitionedCall/Expand1x1Fire6/StatefulPartitionedCall:output:0/Expand3x3Fire6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate6_layer_call_and_return_conditional_losses_2655832
Concatenate6/PartitionedCallา
$SqueezeFire7/StatefulPartitionedCallStatefulPartitionedCall%Concatenate6/PartitionedCall:output:0squeezefire7_265620squeezefire7_265622*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_2656092&
$SqueezeFire7/StatefulPartitionedCallๅ
&Expand1x1Fire7/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire7/StatefulPartitionedCall:output:0expand1x1fire7_265653expand1x1fire7_265655*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_2656422(
&Expand1x1Fire7/StatefulPartitionedCallๅ
&Expand3x3Fire7/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire7/StatefulPartitionedCall:output:0expand3x3fire7_265686expand3x3fire7_265688*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_2656752(
&Expand3x3Fire7/StatefulPartitionedCallว
Concatenate7/PartitionedCallPartitionedCall/Expand1x1Fire7/StatefulPartitionedCall:output:0/Expand3x3Fire7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate7_layer_call_and_return_conditional_losses_2656982
Concatenate7/PartitionedCallา
$SqueezeFire8/StatefulPartitionedCallStatefulPartitionedCall%Concatenate7/PartitionedCall:output:0squeezefire8_265735squeezefire8_265737*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_2657242&
$SqueezeFire8/StatefulPartitionedCallๅ
&Expand1x1Fire8/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire8/StatefulPartitionedCall:output:0expand1x1fire8_265768expand1x1fire8_265770*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_2657572(
&Expand1x1Fire8/StatefulPartitionedCallๅ
&Expand3x3Fire8/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire8/StatefulPartitionedCall:output:0expand3x3fire8_265801expand3x3fire8_265803*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_2657902(
&Expand3x3Fire8/StatefulPartitionedCallว
Concatenate8/PartitionedCallPartitionedCall/Expand1x1Fire8/StatefulPartitionedCall:output:0/Expand3x3Fire8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate8_layer_call_and_return_conditional_losses_2658132
Concatenate8/PartitionedCall?
MaxPool8/PartitionedCallPartitionedCall%Concatenate8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool8_layer_call_and_return_conditional_losses_2649722
MaxPool8/PartitionedCallฮ
$SqueezeFire9/StatefulPartitionedCallStatefulPartitionedCall!MaxPool8/PartitionedCall:output:0squeezefire9_265851squeezefire9_265853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_2658402&
$SqueezeFire9/StatefulPartitionedCallๅ
&Expand1x1Fire9/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire9/StatefulPartitionedCall:output:0expand1x1fire9_265884expand1x1fire9_265886*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_2658732(
&Expand1x1Fire9/StatefulPartitionedCallๅ
&Expand3x3Fire9/StatefulPartitionedCallStatefulPartitionedCall-SqueezeFire9/StatefulPartitionedCall:output:0expand3x3fire9_265917expand3x3fire9_265919*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_2659062(
&Expand3x3Fire9/StatefulPartitionedCallว
Concatenate9/PartitionedCallPartitionedCall/Expand1x1Fire9/StatefulPartitionedCall:output:0/Expand3x3Fire9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate9_layer_call_and_return_conditional_losses_2659292
Concatenate9/PartitionedCall
 Dropout9/StatefulPartitionedCallStatefulPartitionedCall%Concatenate9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dropout9_layer_call_and_return_conditional_losses_2659502"
 Dropout9/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)Dropout9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????ค* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2659742
flatten_1/PartitionedCallฝ
"DenseFinal/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0densefinal_266004densefinal_266006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_DenseFinal_layer_call_and_return_conditional_losses_2659932$
"DenseFinal/StatefulPartitionedCallฌ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_265010*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulด
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOpsqueezefire2_265044*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpฉ
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_1/Square
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constข
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_1/mul/xค
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulถ
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOpexpand1x1fire2_265077*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpฉ
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_2/Square
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constข
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_2/mul/xค
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulถ
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOpexpand3x3fire2_265110*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_3/Square/ReadVariableOpฉ
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_3/Square
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_3/Constข
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/Sum}
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_3/mul/xค
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/mulต
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOpsqueezefire3_265159*'
_output_shapes
:*
dtype02,
*kernel/Regularizer_4/Square/ReadVariableOpช
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer_4/Square
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_4/Constข
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/Sum}
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_4/mul/xค
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/mulถ
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOpexpand1x1fire3_265192*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_5/Square/ReadVariableOpฉ
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_5/Square
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_5/Constข
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/Sum}
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_5/mul/xค
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/mulถ
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOpexpand3x3fire3_265225*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_6/Square/ReadVariableOpฉ
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_6/Square
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_6/Constข
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/Sum}
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_6/mul/xค
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/mulต
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOpsqueezefire4_265274*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_7/Square/ReadVariableOpช
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_7/Square
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_7/Constข
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/Sum}
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_7/mul/xค
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/mulท
*kernel/Regularizer_8/Square/ReadVariableOpReadVariableOpexpand1x1fire4_265307*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_8/Square/ReadVariableOpช
kernel/Regularizer_8/SquareSquare2kernel/Regularizer_8/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_8/Square
kernel/Regularizer_8/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_8/Constข
kernel/Regularizer_8/SumSumkernel/Regularizer_8/Square:y:0#kernel/Regularizer_8/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/Sum}
kernel/Regularizer_8/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_8/mul/xค
kernel/Regularizer_8/mulMul#kernel/Regularizer_8/mul/x:output:0!kernel/Regularizer_8/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/mulท
*kernel/Regularizer_9/Square/ReadVariableOpReadVariableOpexpand3x3fire4_265340*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_9/Square/ReadVariableOpช
kernel/Regularizer_9/SquareSquare2kernel/Regularizer_9/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_9/Square
kernel/Regularizer_9/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_9/Constข
kernel/Regularizer_9/SumSumkernel/Regularizer_9/Square:y:0#kernel/Regularizer_9/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/Sum}
kernel/Regularizer_9/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_9/mul/xค
kernel/Regularizer_9/mulMul#kernel/Regularizer_9/mul/x:output:0!kernel/Regularizer_9/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/mulท
+kernel/Regularizer_10/Square/ReadVariableOpReadVariableOpsqueezefire5_265390*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_10/Square/ReadVariableOpญ
kernel/Regularizer_10/SquareSquare3kernel/Regularizer_10/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_10/Square
kernel/Regularizer_10/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_10/Constฆ
kernel/Regularizer_10/SumSum kernel/Regularizer_10/Square:y:0$kernel/Regularizer_10/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/Sum
kernel/Regularizer_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_10/mul/xจ
kernel/Regularizer_10/mulMul$kernel/Regularizer_10/mul/x:output:0"kernel/Regularizer_10/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/mulน
+kernel/Regularizer_11/Square/ReadVariableOpReadVariableOpexpand1x1fire5_265423*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_11/Square/ReadVariableOpญ
kernel/Regularizer_11/SquareSquare3kernel/Regularizer_11/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_11/Square
kernel/Regularizer_11/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_11/Constฆ
kernel/Regularizer_11/SumSum kernel/Regularizer_11/Square:y:0$kernel/Regularizer_11/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/Sum
kernel/Regularizer_11/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_11/mul/xจ
kernel/Regularizer_11/mulMul$kernel/Regularizer_11/mul/x:output:0"kernel/Regularizer_11/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/mulน
+kernel/Regularizer_12/Square/ReadVariableOpReadVariableOpexpand3x3fire5_265456*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_12/Square/ReadVariableOpญ
kernel/Regularizer_12/SquareSquare3kernel/Regularizer_12/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_12/Square
kernel/Regularizer_12/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_12/Constฆ
kernel/Regularizer_12/SumSum kernel/Regularizer_12/Square:y:0$kernel/Regularizer_12/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/Sum
kernel/Regularizer_12/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_12/mul/xจ
kernel/Regularizer_12/mulMul$kernel/Regularizer_12/mul/x:output:0"kernel/Regularizer_12/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/mulท
+kernel/Regularizer_13/Square/ReadVariableOpReadVariableOpsqueezefire6_265505*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_13/Square/ReadVariableOpญ
kernel/Regularizer_13/SquareSquare3kernel/Regularizer_13/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_13/Square
kernel/Regularizer_13/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_13/Constฆ
kernel/Regularizer_13/SumSum kernel/Regularizer_13/Square:y:0$kernel/Regularizer_13/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/Sum
kernel/Regularizer_13/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_13/mul/xจ
kernel/Regularizer_13/mulMul$kernel/Regularizer_13/mul/x:output:0"kernel/Regularizer_13/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/mulน
+kernel/Regularizer_14/Square/ReadVariableOpReadVariableOpexpand1x1fire6_265538*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_14/Square/ReadVariableOpญ
kernel/Regularizer_14/SquareSquare3kernel/Regularizer_14/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_14/Square
kernel/Regularizer_14/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_14/Constฆ
kernel/Regularizer_14/SumSum kernel/Regularizer_14/Square:y:0$kernel/Regularizer_14/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/Sum
kernel/Regularizer_14/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_14/mul/xจ
kernel/Regularizer_14/mulMul$kernel/Regularizer_14/mul/x:output:0"kernel/Regularizer_14/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/mulน
+kernel/Regularizer_15/Square/ReadVariableOpReadVariableOpexpand3x3fire6_265571*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_15/Square/ReadVariableOpญ
kernel/Regularizer_15/SquareSquare3kernel/Regularizer_15/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_15/Square
kernel/Regularizer_15/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_15/Constฆ
kernel/Regularizer_15/SumSum kernel/Regularizer_15/Square:y:0$kernel/Regularizer_15/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/Sum
kernel/Regularizer_15/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_15/mul/xจ
kernel/Regularizer_15/mulMul$kernel/Regularizer_15/mul/x:output:0"kernel/Regularizer_15/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/mulท
+kernel/Regularizer_16/Square/ReadVariableOpReadVariableOpsqueezefire7_265620*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_16/Square/ReadVariableOpญ
kernel/Regularizer_16/SquareSquare3kernel/Regularizer_16/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_16/Square
kernel/Regularizer_16/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_16/Constฆ
kernel/Regularizer_16/SumSum kernel/Regularizer_16/Square:y:0$kernel/Regularizer_16/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/Sum
kernel/Regularizer_16/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_16/mul/xจ
kernel/Regularizer_16/mulMul$kernel/Regularizer_16/mul/x:output:0"kernel/Regularizer_16/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/mulน
+kernel/Regularizer_17/Square/ReadVariableOpReadVariableOpexpand1x1fire7_265653*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_17/Square/ReadVariableOpญ
kernel/Regularizer_17/SquareSquare3kernel/Regularizer_17/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_17/Square
kernel/Regularizer_17/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_17/Constฆ
kernel/Regularizer_17/SumSum kernel/Regularizer_17/Square:y:0$kernel/Regularizer_17/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/Sum
kernel/Regularizer_17/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_17/mul/xจ
kernel/Regularizer_17/mulMul$kernel/Regularizer_17/mul/x:output:0"kernel/Regularizer_17/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/mulน
+kernel/Regularizer_18/Square/ReadVariableOpReadVariableOpexpand3x3fire7_265686*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_18/Square/ReadVariableOpญ
kernel/Regularizer_18/SquareSquare3kernel/Regularizer_18/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_18/Square
kernel/Regularizer_18/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_18/Constฆ
kernel/Regularizer_18/SumSum kernel/Regularizer_18/Square:y:0$kernel/Regularizer_18/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/Sum
kernel/Regularizer_18/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_18/mul/xจ
kernel/Regularizer_18/mulMul$kernel/Regularizer_18/mul/x:output:0"kernel/Regularizer_18/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/mulท
+kernel/Regularizer_19/Square/ReadVariableOpReadVariableOpsqueezefire8_265735*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_19/Square/ReadVariableOpญ
kernel/Regularizer_19/SquareSquare3kernel/Regularizer_19/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_19/Square
kernel/Regularizer_19/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_19/Constฆ
kernel/Regularizer_19/SumSum kernel/Regularizer_19/Square:y:0$kernel/Regularizer_19/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/Sum
kernel/Regularizer_19/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_19/mul/xจ
kernel/Regularizer_19/mulMul$kernel/Regularizer_19/mul/x:output:0"kernel/Regularizer_19/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/mulน
+kernel/Regularizer_20/Square/ReadVariableOpReadVariableOpexpand1x1fire8_265768*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_20/Square/ReadVariableOpญ
kernel/Regularizer_20/SquareSquare3kernel/Regularizer_20/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_20/Square
kernel/Regularizer_20/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_20/Constฆ
kernel/Regularizer_20/SumSum kernel/Regularizer_20/Square:y:0$kernel/Regularizer_20/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/Sum
kernel/Regularizer_20/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_20/mul/xจ
kernel/Regularizer_20/mulMul$kernel/Regularizer_20/mul/x:output:0"kernel/Regularizer_20/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/mulน
+kernel/Regularizer_21/Square/ReadVariableOpReadVariableOpexpand3x3fire8_265801*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_21/Square/ReadVariableOpญ
kernel/Regularizer_21/SquareSquare3kernel/Regularizer_21/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_21/Square
kernel/Regularizer_21/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_21/Constฆ
kernel/Regularizer_21/SumSum kernel/Regularizer_21/Square:y:0$kernel/Regularizer_21/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/Sum
kernel/Regularizer_21/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_21/mul/xจ
kernel/Regularizer_21/mulMul$kernel/Regularizer_21/mul/x:output:0"kernel/Regularizer_21/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/mulท
+kernel/Regularizer_22/Square/ReadVariableOpReadVariableOpsqueezefire9_265851*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_22/Square/ReadVariableOpญ
kernel/Regularizer_22/SquareSquare3kernel/Regularizer_22/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_22/Square
kernel/Regularizer_22/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_22/Constฆ
kernel/Regularizer_22/SumSum kernel/Regularizer_22/Square:y:0$kernel/Regularizer_22/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/Sum
kernel/Regularizer_22/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_22/mul/xจ
kernel/Regularizer_22/mulMul$kernel/Regularizer_22/mul/x:output:0"kernel/Regularizer_22/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/mulน
+kernel/Regularizer_23/Square/ReadVariableOpReadVariableOpexpand1x1fire9_265884*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_23/Square/ReadVariableOpญ
kernel/Regularizer_23/SquareSquare3kernel/Regularizer_23/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_23/Square
kernel/Regularizer_23/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_23/Constฆ
kernel/Regularizer_23/SumSum kernel/Regularizer_23/Square:y:0$kernel/Regularizer_23/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/Sum
kernel/Regularizer_23/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_23/mul/xจ
kernel/Regularizer_23/mulMul$kernel/Regularizer_23/mul/x:output:0"kernel/Regularizer_23/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/mulน
+kernel/Regularizer_24/Square/ReadVariableOpReadVariableOpexpand3x3fire9_265917*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_24/Square/ReadVariableOpญ
kernel/Regularizer_24/SquareSquare3kernel/Regularizer_24/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_24/Square
kernel/Regularizer_24/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_24/Constฆ
kernel/Regularizer_24/SumSum kernel/Regularizer_24/Square:y:0$kernel/Regularizer_24/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/Sum
kernel/Regularizer_24/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_24/mul/xจ
kernel/Regularizer_24/mulMul$kernel/Regularizer_24/mul/x:output:0"kernel/Regularizer_24/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/mulฒ	
IdentityIdentity+DenseFinal/StatefulPartitionedCall:output:0!^Conv2D_1/StatefulPartitionedCall#^DenseFinal/StatefulPartitionedCall!^Dropout9/StatefulPartitionedCall'^Expand1x1Fire2/StatefulPartitionedCall'^Expand1x1Fire3/StatefulPartitionedCall'^Expand1x1Fire4/StatefulPartitionedCall'^Expand1x1Fire5/StatefulPartitionedCall'^Expand1x1Fire6/StatefulPartitionedCall'^Expand1x1Fire7/StatefulPartitionedCall'^Expand1x1Fire8/StatefulPartitionedCall'^Expand1x1Fire9/StatefulPartitionedCall'^Expand3x3Fire2/StatefulPartitionedCall'^Expand3x3Fire3/StatefulPartitionedCall'^Expand3x3Fire4/StatefulPartitionedCall'^Expand3x3Fire5/StatefulPartitionedCall'^Expand3x3Fire6/StatefulPartitionedCall'^Expand3x3Fire7/StatefulPartitionedCall'^Expand3x3Fire8/StatefulPartitionedCall'^Expand3x3Fire9/StatefulPartitionedCall%^SqueezeFire2/StatefulPartitionedCall%^SqueezeFire3/StatefulPartitionedCall%^SqueezeFire4/StatefulPartitionedCall%^SqueezeFire5/StatefulPartitionedCall%^SqueezeFire6/StatefulPartitionedCall%^SqueezeFire7/StatefulPartitionedCall%^SqueezeFire8/StatefulPartitionedCall%^SqueezeFire9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2H
"DenseFinal/StatefulPartitionedCall"DenseFinal/StatefulPartitionedCall2D
 Dropout9/StatefulPartitionedCall Dropout9/StatefulPartitionedCall2P
&Expand1x1Fire2/StatefulPartitionedCall&Expand1x1Fire2/StatefulPartitionedCall2P
&Expand1x1Fire3/StatefulPartitionedCall&Expand1x1Fire3/StatefulPartitionedCall2P
&Expand1x1Fire4/StatefulPartitionedCall&Expand1x1Fire4/StatefulPartitionedCall2P
&Expand1x1Fire5/StatefulPartitionedCall&Expand1x1Fire5/StatefulPartitionedCall2P
&Expand1x1Fire6/StatefulPartitionedCall&Expand1x1Fire6/StatefulPartitionedCall2P
&Expand1x1Fire7/StatefulPartitionedCall&Expand1x1Fire7/StatefulPartitionedCall2P
&Expand1x1Fire8/StatefulPartitionedCall&Expand1x1Fire8/StatefulPartitionedCall2P
&Expand1x1Fire9/StatefulPartitionedCall&Expand1x1Fire9/StatefulPartitionedCall2P
&Expand3x3Fire2/StatefulPartitionedCall&Expand3x3Fire2/StatefulPartitionedCall2P
&Expand3x3Fire3/StatefulPartitionedCall&Expand3x3Fire3/StatefulPartitionedCall2P
&Expand3x3Fire4/StatefulPartitionedCall&Expand3x3Fire4/StatefulPartitionedCall2P
&Expand3x3Fire5/StatefulPartitionedCall&Expand3x3Fire5/StatefulPartitionedCall2P
&Expand3x3Fire6/StatefulPartitionedCall&Expand3x3Fire6/StatefulPartitionedCall2P
&Expand3x3Fire7/StatefulPartitionedCall&Expand3x3Fire7/StatefulPartitionedCall2P
&Expand3x3Fire8/StatefulPartitionedCall&Expand3x3Fire8/StatefulPartitionedCall2P
&Expand3x3Fire9/StatefulPartitionedCall&Expand3x3Fire9/StatefulPartitionedCall2L
$SqueezeFire2/StatefulPartitionedCall$SqueezeFire2/StatefulPartitionedCall2L
$SqueezeFire3/StatefulPartitionedCall$SqueezeFire3/StatefulPartitionedCall2L
$SqueezeFire4/StatefulPartitionedCall$SqueezeFire4/StatefulPartitionedCall2L
$SqueezeFire5/StatefulPartitionedCall$SqueezeFire5/StatefulPartitionedCall2L
$SqueezeFire6/StatefulPartitionedCall$SqueezeFire6/StatefulPartitionedCall2L
$SqueezeFire7/StatefulPartitionedCall$SqueezeFire7/StatefulPartitionedCall2L
$SqueezeFire8/StatefulPartitionedCall$SqueezeFire8/StatefulPartitionedCall2L
$SqueezeFire9/StatefulPartitionedCall$SqueezeFire9/StatefulPartitionedCall:X T
1
_output_shapes
:?????????เเ

_user_specified_nameInput

?
5__inference_SqueezeNet_Preloaded_layer_call_fn_266864	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identityขStatefulPartitionedCallต
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_2667572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:?????????เเ

_user_specified_nameInput
ย
ฒ
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_269219

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs


-__inference_SqueezeFire2_layer_call_fn_268542

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????77*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_2650332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????77@
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_268751

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77 2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????77:::X T
0
_output_shapes
:?????????77
 
_user_specified_nameinputs
ก
ก
5__inference_SqueezeNet_Preloaded_layer_call_fn_268478

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identityขStatefulPartitionedCallถ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_2671632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_265790

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_269078

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????02
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_265642

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_265445

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
๓
t
H__inference_Concatenate6_layer_call_and_return_conditional_losses_269049
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????ภ:?????????ภ:Z V
0
_output_shapes
:?????????ภ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????ภ
"
_user_specified_name
inputs/1
ฝ	
f
__inference_loss_fn_3_2694845
1kernel_regularizer_square_readvariableop_resource
identityฮ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
๓
t
H__inference_Concatenate7_layer_call_and_return_conditional_losses_269158
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????ภ:?????????ภ:Z V
0
_output_shapes
:?????????ภ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????ภ
"
_user_specified_name
inputs/1


/__inference_Expand1x1Fire6_layer_call_fn_269010

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_2655272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
ฝ
ฐ
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_269187

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ุ
Y
-__inference_Concatenate4_layer_call_fn_268837
inputs_0
inputs_1
identity฿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????77* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Concatenate4_layer_call_and_return_conditional_losses_2653522
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????77:?????????77:Z V
0
_output_shapes
:?????????77
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????77
"
_user_specified_name
inputs/1
ข
E
)__inference_MaxPool1_layer_call_fn_264954

inputs
identity่
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_MaxPool1_layer_call_and_return_conditional_losses_2649482
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ฝ	
f
__inference_loss_fn_6_2695175
1kernel_regularizer_square_readvariableop_resource
identityฮ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ส
c
D__inference_Dropout9_layer_call_and_return_conditional_losses_269394

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeฝ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yว
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


-__inference_SqueezeFire6_layer_call_fn_268978

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_2654942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


-__inference_SqueezeFire5_layer_call_fn_268869

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_2653792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ฝ	
f
__inference_loss_fn_2_2694735
1kernel_regularizer_square_readvariableop_resource
identityฮ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ธ
ฐ
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_265033

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????772
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????772

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77@:::W S
/
_output_shapes
:?????????77@
 
_user_specified_nameinputs
ย
ฒ
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_265675

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
๋
r
H__inference_Concatenate8_layer_call_and_return_conditional_losses_265813

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs


/__inference_Expand1x1Fire8_layer_call_fn_269228

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_2657572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
บ
ฒ
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_265066

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77:::W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
บ
ฒ
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_268565

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77:::W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
บ
ฒ
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_268706

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77:::W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
น
ฎ
F__inference_DenseFinal_layer_call_and_return_conditional_losses_269431

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ค*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:?????????ค:::Q M
)
_output_shapes
:?????????ค
 
_user_specified_nameinputs
ๆน
ฑ
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_268260

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource/
+squeezefire2_conv2d_readvariableop_resource0
,squeezefire2_biasadd_readvariableop_resource1
-expand1x1fire2_conv2d_readvariableop_resource2
.expand1x1fire2_biasadd_readvariableop_resource1
-expand3x3fire2_conv2d_readvariableop_resource2
.expand3x3fire2_biasadd_readvariableop_resource/
+squeezefire3_conv2d_readvariableop_resource0
,squeezefire3_biasadd_readvariableop_resource1
-expand1x1fire3_conv2d_readvariableop_resource2
.expand1x1fire3_biasadd_readvariableop_resource1
-expand3x3fire3_conv2d_readvariableop_resource2
.expand3x3fire3_biasadd_readvariableop_resource/
+squeezefire4_conv2d_readvariableop_resource0
,squeezefire4_biasadd_readvariableop_resource1
-expand1x1fire4_conv2d_readvariableop_resource2
.expand1x1fire4_biasadd_readvariableop_resource1
-expand3x3fire4_conv2d_readvariableop_resource2
.expand3x3fire4_biasadd_readvariableop_resource/
+squeezefire5_conv2d_readvariableop_resource0
,squeezefire5_biasadd_readvariableop_resource1
-expand1x1fire5_conv2d_readvariableop_resource2
.expand1x1fire5_biasadd_readvariableop_resource1
-expand3x3fire5_conv2d_readvariableop_resource2
.expand3x3fire5_biasadd_readvariableop_resource/
+squeezefire6_conv2d_readvariableop_resource0
,squeezefire6_biasadd_readvariableop_resource1
-expand1x1fire6_conv2d_readvariableop_resource2
.expand1x1fire6_biasadd_readvariableop_resource1
-expand3x3fire6_conv2d_readvariableop_resource2
.expand3x3fire6_biasadd_readvariableop_resource/
+squeezefire7_conv2d_readvariableop_resource0
,squeezefire7_biasadd_readvariableop_resource1
-expand1x1fire7_conv2d_readvariableop_resource2
.expand1x1fire7_biasadd_readvariableop_resource1
-expand3x3fire7_conv2d_readvariableop_resource2
.expand3x3fire7_biasadd_readvariableop_resource/
+squeezefire8_conv2d_readvariableop_resource0
,squeezefire8_biasadd_readvariableop_resource1
-expand1x1fire8_conv2d_readvariableop_resource2
.expand1x1fire8_biasadd_readvariableop_resource1
-expand3x3fire8_conv2d_readvariableop_resource2
.expand3x3fire8_biasadd_readvariableop_resource/
+squeezefire9_conv2d_readvariableop_resource0
,squeezefire9_biasadd_readvariableop_resource1
-expand1x1fire9_conv2d_readvariableop_resource2
.expand1x1fire9_biasadd_readvariableop_resource1
-expand3x3fire9_conv2d_readvariableop_resource2
.expand3x3fire9_biasadd_readvariableop_resource-
)densefinal_matmul_readvariableop_resource.
*densefinal_biasadd_readvariableop_resource
identityฐ
Conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
Conv2D_1/Conv2D/ReadVariableOpพ
Conv2D_1/Conv2DConv2Dinputs&Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@*
paddingSAME*
strides
2
Conv2D_1/Conv2Dง
Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
Conv2D_1/BiasAdd/ReadVariableOpฌ
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2D:output:0'Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????pp@2
Conv2D_1/BiasAdd{
Conv2D_1/ReluReluConv2D_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????pp@2
Conv2D_1/Reluน
MaxPool1/MaxPoolMaxPoolConv2D_1/Relu:activations:0*/
_output_shapes
:?????????77@*
ksize
*
paddingVALID*
strides
2
MaxPool1/MaxPoolผ
"SqueezeFire2/Conv2D/ReadVariableOpReadVariableOp+squeezefire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"SqueezeFire2/Conv2D/ReadVariableOp?
SqueezeFire2/Conv2DConv2DMaxPool1/MaxPool:output:0*SqueezeFire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2
SqueezeFire2/Conv2Dณ
#SqueezeFire2/BiasAdd/ReadVariableOpReadVariableOp,squeezefire2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#SqueezeFire2/BiasAdd/ReadVariableOpผ
SqueezeFire2/BiasAddBiasAddSqueezeFire2/Conv2D:output:0+SqueezeFire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772
SqueezeFire2/BiasAdd
SqueezeFire2/ReluReluSqueezeFire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????772
SqueezeFire2/Reluย
$Expand1x1Fire2/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Expand1x1Fire2/Conv2D/ReadVariableOp้
Expand1x1Fire2/Conv2DConv2DSqueezeFire2/Relu:activations:0,Expand1x1Fire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Expand1x1Fire2/Conv2Dน
%Expand1x1Fire2/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%Expand1x1Fire2/BiasAdd/ReadVariableOpฤ
Expand1x1Fire2/BiasAddBiasAddExpand1x1Fire2/Conv2D:output:0-Expand1x1Fire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2
Expand1x1Fire2/BiasAdd
Expand1x1Fire2/ReluReluExpand1x1Fire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Expand1x1Fire2/Reluย
$Expand3x3Fire2/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Expand3x3Fire2/Conv2D/ReadVariableOp้
Expand3x3Fire2/Conv2DConv2DSqueezeFire2/Relu:activations:0,Expand3x3Fire2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Expand3x3Fire2/Conv2Dน
%Expand3x3Fire2/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%Expand3x3Fire2/BiasAdd/ReadVariableOpฤ
Expand3x3Fire2/BiasAddBiasAddExpand3x3Fire2/Conv2D:output:0-Expand3x3Fire2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2
Expand3x3Fire2/BiasAdd
Expand3x3Fire2/ReluReluExpand3x3Fire2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Expand3x3Fire2/Reluv
Concatenate2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate2/concat/axisใ
Concatenate2/concatConcatV2!Expand1x1Fire2/Relu:activations:0!Expand3x3Fire2/Relu:activations:0!Concatenate2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
Concatenate2/concatฝ
"SqueezeFire3/Conv2D/ReadVariableOpReadVariableOp+squeezefire3_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02$
"SqueezeFire3/Conv2D/ReadVariableOpเ
SqueezeFire3/Conv2DConv2DConcatenate2/concat:output:0*SqueezeFire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77*
paddingSAME*
strides
2
SqueezeFire3/Conv2Dณ
#SqueezeFire3/BiasAdd/ReadVariableOpReadVariableOp,squeezefire3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#SqueezeFire3/BiasAdd/ReadVariableOpผ
SqueezeFire3/BiasAddBiasAddSqueezeFire3/Conv2D:output:0+SqueezeFire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????772
SqueezeFire3/BiasAdd
SqueezeFire3/ReluReluSqueezeFire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????772
SqueezeFire3/Reluย
$Expand1x1Fire3/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Expand1x1Fire3/Conv2D/ReadVariableOp้
Expand1x1Fire3/Conv2DConv2DSqueezeFire3/Relu:activations:0,Expand1x1Fire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Expand1x1Fire3/Conv2Dน
%Expand1x1Fire3/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%Expand1x1Fire3/BiasAdd/ReadVariableOpฤ
Expand1x1Fire3/BiasAddBiasAddExpand1x1Fire3/Conv2D:output:0-Expand1x1Fire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2
Expand1x1Fire3/BiasAdd
Expand1x1Fire3/ReluReluExpand1x1Fire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Expand1x1Fire3/Reluย
$Expand3x3Fire3/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Expand3x3Fire3/Conv2D/ReadVariableOp้
Expand3x3Fire3/Conv2DConv2DSqueezeFire3/Relu:activations:0,Expand3x3Fire3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Expand3x3Fire3/Conv2Dน
%Expand3x3Fire3/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%Expand3x3Fire3/BiasAdd/ReadVariableOpฤ
Expand3x3Fire3/BiasAddBiasAddExpand3x3Fire3/Conv2D:output:0-Expand3x3Fire3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2
Expand3x3Fire3/BiasAdd
Expand3x3Fire3/ReluReluExpand3x3Fire3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Expand3x3Fire3/Reluv
Concatenate3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate3/concat/axisใ
Concatenate3/concatConcatV2!Expand1x1Fire3/Relu:activations:0!Expand3x3Fire3/Relu:activations:0!Concatenate3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
Concatenate3/concatฝ
"SqueezeFire4/Conv2D/ReadVariableOpReadVariableOp+squeezefire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02$
"SqueezeFire4/Conv2D/ReadVariableOpเ
SqueezeFire4/Conv2DConv2DConcatenate3/concat:output:0*SqueezeFire4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 *
paddingSAME*
strides
2
SqueezeFire4/Conv2Dณ
#SqueezeFire4/BiasAdd/ReadVariableOpReadVariableOp,squeezefire4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#SqueezeFire4/BiasAdd/ReadVariableOpผ
SqueezeFire4/BiasAddBiasAddSqueezeFire4/Conv2D:output:0+SqueezeFire4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77 2
SqueezeFire4/BiasAdd
SqueezeFire4/ReluReluSqueezeFire4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????77 2
SqueezeFire4/Reluร
$Expand1x1Fire4/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02&
$Expand1x1Fire4/Conv2D/ReadVariableOp๊
Expand1x1Fire4/Conv2DConv2DSqueezeFire4/Relu:activations:0,Expand1x1Fire4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2
Expand1x1Fire4/Conv2Dบ
%Expand1x1Fire4/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand1x1Fire4/BiasAdd/ReadVariableOpล
Expand1x1Fire4/BiasAddBiasAddExpand1x1Fire4/Conv2D:output:0-Expand1x1Fire4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772
Expand1x1Fire4/BiasAdd
Expand1x1Fire4/ReluReluExpand1x1Fire4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????772
Expand1x1Fire4/Reluร
$Expand3x3Fire4/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02&
$Expand3x3Fire4/Conv2D/ReadVariableOp๊
Expand3x3Fire4/Conv2DConv2DSqueezeFire4/Relu:activations:0,Expand3x3Fire4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????77*
paddingSAME*
strides
2
Expand3x3Fire4/Conv2Dบ
%Expand3x3Fire4/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand3x3Fire4/BiasAdd/ReadVariableOpล
Expand3x3Fire4/BiasAddBiasAddExpand3x3Fire4/Conv2D:output:0-Expand3x3Fire4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????772
Expand3x3Fire4/BiasAdd
Expand3x3Fire4/ReluReluExpand3x3Fire4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????772
Expand3x3Fire4/Reluv
Concatenate4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate4/concat/axisใ
Concatenate4/concatConcatV2!Expand1x1Fire4/Relu:activations:0!Expand3x3Fire4/Relu:activations:0!Concatenate4/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????772
Concatenate4/concatป
MaxPool4/MaxPoolMaxPoolConcatenate4/concat:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
MaxPool4/MaxPoolฝ
"SqueezeFire5/Conv2D/ReadVariableOpReadVariableOp+squeezefire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02$
"SqueezeFire5/Conv2D/ReadVariableOp?
SqueezeFire5/Conv2DConv2DMaxPool4/MaxPool:output:0*SqueezeFire5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
SqueezeFire5/Conv2Dณ
#SqueezeFire5/BiasAdd/ReadVariableOpReadVariableOp,squeezefire5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#SqueezeFire5/BiasAdd/ReadVariableOpผ
SqueezeFire5/BiasAddBiasAddSqueezeFire5/Conv2D:output:0+SqueezeFire5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
SqueezeFire5/BiasAdd
SqueezeFire5/ReluReluSqueezeFire5/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
SqueezeFire5/Reluร
$Expand1x1Fire5/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02&
$Expand1x1Fire5/Conv2D/ReadVariableOp๊
Expand1x1Fire5/Conv2DConv2DSqueezeFire5/Relu:activations:0,Expand1x1Fire5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand1x1Fire5/Conv2Dบ
%Expand1x1Fire5/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand1x1Fire5/BiasAdd/ReadVariableOpล
Expand1x1Fire5/BiasAddBiasAddExpand1x1Fire5/Conv2D:output:0-Expand1x1Fire5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire5/BiasAdd
Expand1x1Fire5/ReluReluExpand1x1Fire5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire5/Reluร
$Expand3x3Fire5/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02&
$Expand3x3Fire5/Conv2D/ReadVariableOp๊
Expand3x3Fire5/Conv2DConv2DSqueezeFire5/Relu:activations:0,Expand3x3Fire5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand3x3Fire5/Conv2Dบ
%Expand3x3Fire5/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand3x3Fire5/BiasAdd/ReadVariableOpล
Expand3x3Fire5/BiasAddBiasAddExpand3x3Fire5/Conv2D:output:0-Expand3x3Fire5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire5/BiasAdd
Expand3x3Fire5/ReluReluExpand3x3Fire5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire5/Reluv
Concatenate5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate5/concat/axisใ
Concatenate5/concatConcatV2!Expand1x1Fire5/Relu:activations:0!Expand3x3Fire5/Relu:activations:0!Concatenate5/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate5/concatฝ
"SqueezeFire6/Conv2D/ReadVariableOpReadVariableOp+squeezefire6_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02$
"SqueezeFire6/Conv2D/ReadVariableOpเ
SqueezeFire6/Conv2DConv2DConcatenate5/concat:output:0*SqueezeFire6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2
SqueezeFire6/Conv2Dณ
#SqueezeFire6/BiasAdd/ReadVariableOpReadVariableOp,squeezefire6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02%
#SqueezeFire6/BiasAdd/ReadVariableOpผ
SqueezeFire6/BiasAddBiasAddSqueezeFire6/Conv2D:output:0+SqueezeFire6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
SqueezeFire6/BiasAdd
SqueezeFire6/ReluReluSqueezeFire6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
SqueezeFire6/Reluร
$Expand1x1Fire6/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02&
$Expand1x1Fire6/Conv2D/ReadVariableOp๊
Expand1x1Fire6/Conv2DConv2DSqueezeFire6/Relu:activations:0,Expand1x1Fire6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Expand1x1Fire6/Conv2Dบ
%Expand1x1Fire6/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire6_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02'
%Expand1x1Fire6/BiasAdd/ReadVariableOpล
Expand1x1Fire6/BiasAddBiasAddExpand1x1Fire6/Conv2D:output:0-Expand1x1Fire6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2
Expand1x1Fire6/BiasAdd
Expand1x1Fire6/ReluReluExpand1x1Fire6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Expand1x1Fire6/Reluร
$Expand3x3Fire6/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02&
$Expand3x3Fire6/Conv2D/ReadVariableOp๊
Expand3x3Fire6/Conv2DConv2DSqueezeFire6/Relu:activations:0,Expand3x3Fire6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Expand3x3Fire6/Conv2Dบ
%Expand3x3Fire6/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire6_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02'
%Expand3x3Fire6/BiasAdd/ReadVariableOpล
Expand3x3Fire6/BiasAddBiasAddExpand3x3Fire6/Conv2D:output:0-Expand3x3Fire6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2
Expand3x3Fire6/BiasAdd
Expand3x3Fire6/ReluReluExpand3x3Fire6/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Expand3x3Fire6/Reluv
Concatenate6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate6/concat/axisใ
Concatenate6/concatConcatV2!Expand1x1Fire6/Relu:activations:0!Expand3x3Fire6/Relu:activations:0!Concatenate6/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate6/concatฝ
"SqueezeFire7/Conv2D/ReadVariableOpReadVariableOp+squeezefire7_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02$
"SqueezeFire7/Conv2D/ReadVariableOpเ
SqueezeFire7/Conv2DConv2DConcatenate6/concat:output:0*SqueezeFire7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2
SqueezeFire7/Conv2Dณ
#SqueezeFire7/BiasAdd/ReadVariableOpReadVariableOp,squeezefire7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02%
#SqueezeFire7/BiasAdd/ReadVariableOpผ
SqueezeFire7/BiasAddBiasAddSqueezeFire7/Conv2D:output:0+SqueezeFire7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
SqueezeFire7/BiasAdd
SqueezeFire7/ReluReluSqueezeFire7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
SqueezeFire7/Reluร
$Expand1x1Fire7/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02&
$Expand1x1Fire7/Conv2D/ReadVariableOp๊
Expand1x1Fire7/Conv2DConv2DSqueezeFire7/Relu:activations:0,Expand1x1Fire7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Expand1x1Fire7/Conv2Dบ
%Expand1x1Fire7/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire7_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02'
%Expand1x1Fire7/BiasAdd/ReadVariableOpล
Expand1x1Fire7/BiasAddBiasAddExpand1x1Fire7/Conv2D:output:0-Expand1x1Fire7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2
Expand1x1Fire7/BiasAdd
Expand1x1Fire7/ReluReluExpand1x1Fire7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Expand1x1Fire7/Reluร
$Expand3x3Fire7/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02&
$Expand3x3Fire7/Conv2D/ReadVariableOp๊
Expand3x3Fire7/Conv2DConv2DSqueezeFire7/Relu:activations:0,Expand3x3Fire7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ*
paddingSAME*
strides
2
Expand3x3Fire7/Conv2Dบ
%Expand3x3Fire7/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire7_biasadd_readvariableop_resource*
_output_shapes	
:ภ*
dtype02'
%Expand3x3Fire7/BiasAdd/ReadVariableOpล
Expand3x3Fire7/BiasAddBiasAddExpand3x3Fire7/Conv2D:output:0-Expand3x3Fire7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????ภ2
Expand3x3Fire7/BiasAdd
Expand3x3Fire7/ReluReluExpand3x3Fire7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ภ2
Expand3x3Fire7/Reluv
Concatenate7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate7/concat/axisใ
Concatenate7/concatConcatV2!Expand1x1Fire7/Relu:activations:0!Expand3x3Fire7/Relu:activations:0!Concatenate7/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate7/concatฝ
"SqueezeFire8/Conv2D/ReadVariableOpReadVariableOp+squeezefire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"SqueezeFire8/Conv2D/ReadVariableOpเ
SqueezeFire8/Conv2DConv2DConcatenate7/concat:output:0*SqueezeFire8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
SqueezeFire8/Conv2Dณ
#SqueezeFire8/BiasAdd/ReadVariableOpReadVariableOp,squeezefire8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#SqueezeFire8/BiasAdd/ReadVariableOpผ
SqueezeFire8/BiasAddBiasAddSqueezeFire8/Conv2D:output:0+SqueezeFire8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
SqueezeFire8/BiasAdd
SqueezeFire8/ReluReluSqueezeFire8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
SqueezeFire8/Reluร
$Expand1x1Fire8/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02&
$Expand1x1Fire8/Conv2D/ReadVariableOp๊
Expand1x1Fire8/Conv2DConv2DSqueezeFire8/Relu:activations:0,Expand1x1Fire8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand1x1Fire8/Conv2Dบ
%Expand1x1Fire8/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand1x1Fire8/BiasAdd/ReadVariableOpล
Expand1x1Fire8/BiasAddBiasAddExpand1x1Fire8/Conv2D:output:0-Expand1x1Fire8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire8/BiasAdd
Expand1x1Fire8/ReluReluExpand1x1Fire8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire8/Reluร
$Expand3x3Fire8/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02&
$Expand3x3Fire8/Conv2D/ReadVariableOp๊
Expand3x3Fire8/Conv2DConv2DSqueezeFire8/Relu:activations:0,Expand3x3Fire8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand3x3Fire8/Conv2Dบ
%Expand3x3Fire8/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand3x3Fire8/BiasAdd/ReadVariableOpล
Expand3x3Fire8/BiasAddBiasAddExpand3x3Fire8/Conv2D:output:0-Expand3x3Fire8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire8/BiasAdd
Expand3x3Fire8/ReluReluExpand3x3Fire8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire8/Reluv
Concatenate8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate8/concat/axisใ
Concatenate8/concatConcatV2!Expand1x1Fire8/Relu:activations:0!Expand3x3Fire8/Relu:activations:0!Concatenate8/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate8/concatป
MaxPool8/MaxPoolMaxPoolConcatenate8/concat:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
MaxPool8/MaxPoolฝ
"SqueezeFire9/Conv2D/ReadVariableOpReadVariableOp+squeezefire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"SqueezeFire9/Conv2D/ReadVariableOp?
SqueezeFire9/Conv2DConv2DMaxPool8/MaxPool:output:0*SqueezeFire9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
SqueezeFire9/Conv2Dณ
#SqueezeFire9/BiasAdd/ReadVariableOpReadVariableOp,squeezefire9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#SqueezeFire9/BiasAdd/ReadVariableOpผ
SqueezeFire9/BiasAddBiasAddSqueezeFire9/Conv2D:output:0+SqueezeFire9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
SqueezeFire9/BiasAdd
SqueezeFire9/ReluReluSqueezeFire9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
SqueezeFire9/Reluร
$Expand1x1Fire9/Conv2D/ReadVariableOpReadVariableOp-expand1x1fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02&
$Expand1x1Fire9/Conv2D/ReadVariableOp๊
Expand1x1Fire9/Conv2DConv2DSqueezeFire9/Relu:activations:0,Expand1x1Fire9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand1x1Fire9/Conv2Dบ
%Expand1x1Fire9/BiasAdd/ReadVariableOpReadVariableOp.expand1x1fire9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand1x1Fire9/BiasAdd/ReadVariableOpล
Expand1x1Fire9/BiasAddBiasAddExpand1x1Fire9/Conv2D:output:0-Expand1x1Fire9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire9/BiasAdd
Expand1x1Fire9/ReluReluExpand1x1Fire9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand1x1Fire9/Reluร
$Expand3x3Fire9/Conv2D/ReadVariableOpReadVariableOp-expand3x3fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02&
$Expand3x3Fire9/Conv2D/ReadVariableOp๊
Expand3x3Fire9/Conv2DConv2DSqueezeFire9/Relu:activations:0,Expand3x3Fire9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Expand3x3Fire9/Conv2Dบ
%Expand3x3Fire9/BiasAdd/ReadVariableOpReadVariableOp.expand3x3fire9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%Expand3x3Fire9/BiasAdd/ReadVariableOpล
Expand3x3Fire9/BiasAddBiasAddExpand3x3Fire9/Conv2D:output:0-Expand3x3Fire9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire9/BiasAdd
Expand3x3Fire9/ReluReluExpand3x3Fire9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Expand3x3Fire9/Reluv
Concatenate9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate9/concat/axisใ
Concatenate9/concatConcatV2!Expand1x1Fire9/Relu:activations:0!Expand3x3Fire9/Relu:activations:0!Concatenate9/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????2
Concatenate9/concat
Dropout9/IdentityIdentityConcatenate9/concat:output:0*
T0*0
_output_shapes
:?????????2
Dropout9/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? R 2
flatten_1/Const
flatten_1/ReshapeReshapeDropout9/Identity:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:?????????ค2
flatten_1/Reshapeฐ
 DenseFinal/MatMul/ReadVariableOpReadVariableOp)densefinal_matmul_readvariableop_resource* 
_output_shapes
:
ค*
dtype02"
 DenseFinal/MatMul/ReadVariableOpจ
DenseFinal/MatMulMatMulflatten_1/Reshape:output:0(DenseFinal/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DenseFinal/MatMulญ
!DenseFinal/BiasAdd/ReadVariableOpReadVariableOp*densefinal_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!DenseFinal/BiasAdd/ReadVariableOpญ
DenseFinal/BiasAddBiasAddDenseFinal/MatMul:product:0)DenseFinal/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DenseFinal/BiasAdd
DenseFinal/SoftmaxSoftmaxDenseFinal/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
DenseFinal/Softmaxฤ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulฬ
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp+squeezefire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpฉ
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_1/Square
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constข
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_1/mul/xค
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulฮ
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp-expand1x1fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpฉ
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_2/Square
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constข
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_2/mul/xค
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulฮ
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOp-expand3x3fire2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_3/Square/ReadVariableOpฉ
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_3/Square
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_3/Constข
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/Sum}
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_3/mul/xค
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_3/mulอ
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOp+squeezefire3_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02,
*kernel/Regularizer_4/Square/ReadVariableOpช
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
kernel/Regularizer_4/Square
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_4/Constข
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/Sum}
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_4/mul/xค
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_4/mulฮ
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOp-expand1x1fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_5/Square/ReadVariableOpฉ
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_5/Square
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_5/Constข
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/Sum}
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_5/mul/xค
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_5/mulฮ
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOp-expand3x3fire3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*kernel/Regularizer_6/Square/ReadVariableOpฉ
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer_6/Square
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_6/Constข
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/Sum}
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_6/mul/xค
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_6/mulอ
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOp+squeezefire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_7/Square/ReadVariableOpช
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_7/Square
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_7/Constข
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/Sum}
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_7/mul/xค
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_7/mulฯ
*kernel/Regularizer_8/Square/ReadVariableOpReadVariableOp-expand1x1fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_8/Square/ReadVariableOpช
kernel/Regularizer_8/SquareSquare2kernel/Regularizer_8/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_8/Square
kernel/Regularizer_8/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_8/Constข
kernel/Regularizer_8/SumSumkernel/Regularizer_8/Square:y:0#kernel/Regularizer_8/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/Sum}
kernel/Regularizer_8/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_8/mul/xค
kernel/Regularizer_8/mulMul#kernel/Regularizer_8/mul/x:output:0!kernel/Regularizer_8/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_8/mulฯ
*kernel/Regularizer_9/Square/ReadVariableOpReadVariableOp-expand3x3fire4_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02,
*kernel/Regularizer_9/Square/ReadVariableOpช
kernel/Regularizer_9/SquareSquare2kernel/Regularizer_9/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_9/Square
kernel/Regularizer_9/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_9/Constข
kernel/Regularizer_9/SumSumkernel/Regularizer_9/Square:y:0#kernel/Regularizer_9/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/Sum}
kernel/Regularizer_9/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_9/mul/xค
kernel/Regularizer_9/mulMul#kernel/Regularizer_9/mul/x:output:0!kernel/Regularizer_9/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_9/mulฯ
+kernel/Regularizer_10/Square/ReadVariableOpReadVariableOp+squeezefire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_10/Square/ReadVariableOpญ
kernel/Regularizer_10/SquareSquare3kernel/Regularizer_10/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_10/Square
kernel/Regularizer_10/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_10/Constฆ
kernel/Regularizer_10/SumSum kernel/Regularizer_10/Square:y:0$kernel/Regularizer_10/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/Sum
kernel/Regularizer_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_10/mul/xจ
kernel/Regularizer_10/mulMul$kernel/Regularizer_10/mul/x:output:0"kernel/Regularizer_10/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_10/mulั
+kernel/Regularizer_11/Square/ReadVariableOpReadVariableOp-expand1x1fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_11/Square/ReadVariableOpญ
kernel/Regularizer_11/SquareSquare3kernel/Regularizer_11/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_11/Square
kernel/Regularizer_11/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_11/Constฆ
kernel/Regularizer_11/SumSum kernel/Regularizer_11/Square:y:0$kernel/Regularizer_11/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/Sum
kernel/Regularizer_11/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_11/mul/xจ
kernel/Regularizer_11/mulMul$kernel/Regularizer_11/mul/x:output:0"kernel/Regularizer_11/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_11/mulั
+kernel/Regularizer_12/Square/ReadVariableOpReadVariableOp-expand3x3fire5_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype02-
+kernel/Regularizer_12/Square/ReadVariableOpญ
kernel/Regularizer_12/SquareSquare3kernel/Regularizer_12/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer_12/Square
kernel/Regularizer_12/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_12/Constฆ
kernel/Regularizer_12/SumSum kernel/Regularizer_12/Square:y:0$kernel/Regularizer_12/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/Sum
kernel/Regularizer_12/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_12/mul/xจ
kernel/Regularizer_12/mulMul$kernel/Regularizer_12/mul/x:output:0"kernel/Regularizer_12/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_12/mulฯ
+kernel/Regularizer_13/Square/ReadVariableOpReadVariableOp+squeezefire6_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_13/Square/ReadVariableOpญ
kernel/Regularizer_13/SquareSquare3kernel/Regularizer_13/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_13/Square
kernel/Regularizer_13/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_13/Constฆ
kernel/Regularizer_13/SumSum kernel/Regularizer_13/Square:y:0$kernel/Regularizer_13/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/Sum
kernel/Regularizer_13/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_13/mul/xจ
kernel/Regularizer_13/mulMul$kernel/Regularizer_13/mul/x:output:0"kernel/Regularizer_13/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_13/mulั
+kernel/Regularizer_14/Square/ReadVariableOpReadVariableOp-expand1x1fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_14/Square/ReadVariableOpญ
kernel/Regularizer_14/SquareSquare3kernel/Regularizer_14/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_14/Square
kernel/Regularizer_14/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_14/Constฆ
kernel/Regularizer_14/SumSum kernel/Regularizer_14/Square:y:0$kernel/Regularizer_14/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/Sum
kernel/Regularizer_14/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_14/mul/xจ
kernel/Regularizer_14/mulMul$kernel/Regularizer_14/mul/x:output:0"kernel/Regularizer_14/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_14/mulั
+kernel/Regularizer_15/Square/ReadVariableOpReadVariableOp-expand3x3fire6_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_15/Square/ReadVariableOpญ
kernel/Regularizer_15/SquareSquare3kernel/Regularizer_15/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_15/Square
kernel/Regularizer_15/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_15/Constฆ
kernel/Regularizer_15/SumSum kernel/Regularizer_15/Square:y:0$kernel/Regularizer_15/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/Sum
kernel/Regularizer_15/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_15/mul/xจ
kernel/Regularizer_15/mulMul$kernel/Regularizer_15/mul/x:output:0"kernel/Regularizer_15/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_15/mulฯ
+kernel/Regularizer_16/Square/ReadVariableOpReadVariableOp+squeezefire7_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02-
+kernel/Regularizer_16/Square/ReadVariableOpญ
kernel/Regularizer_16/SquareSquare3kernel/Regularizer_16/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:02
kernel/Regularizer_16/Square
kernel/Regularizer_16/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_16/Constฆ
kernel/Regularizer_16/SumSum kernel/Regularizer_16/Square:y:0$kernel/Regularizer_16/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/Sum
kernel/Regularizer_16/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_16/mul/xจ
kernel/Regularizer_16/mulMul$kernel/Regularizer_16/mul/x:output:0"kernel/Regularizer_16/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_16/mulั
+kernel/Regularizer_17/Square/ReadVariableOpReadVariableOp-expand1x1fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_17/Square/ReadVariableOpญ
kernel/Regularizer_17/SquareSquare3kernel/Regularizer_17/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_17/Square
kernel/Regularizer_17/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_17/Constฆ
kernel/Regularizer_17/SumSum kernel/Regularizer_17/Square:y:0$kernel/Regularizer_17/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/Sum
kernel/Regularizer_17/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_17/mul/xจ
kernel/Regularizer_17/mulMul$kernel/Regularizer_17/mul/x:output:0"kernel/Regularizer_17/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_17/mulั
+kernel/Regularizer_18/Square/ReadVariableOpReadVariableOp-expand3x3fire7_conv2d_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02-
+kernel/Regularizer_18/Square/ReadVariableOpญ
kernel/Regularizer_18/SquareSquare3kernel/Regularizer_18/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer_18/Square
kernel/Regularizer_18/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_18/Constฆ
kernel/Regularizer_18/SumSum kernel/Regularizer_18/Square:y:0$kernel/Regularizer_18/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/Sum
kernel/Regularizer_18/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_18/mul/xจ
kernel/Regularizer_18/mulMul$kernel/Regularizer_18/mul/x:output:0"kernel/Regularizer_18/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_18/mulฯ
+kernel/Regularizer_19/Square/ReadVariableOpReadVariableOp+squeezefire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_19/Square/ReadVariableOpญ
kernel/Regularizer_19/SquareSquare3kernel/Regularizer_19/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_19/Square
kernel/Regularizer_19/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_19/Constฆ
kernel/Regularizer_19/SumSum kernel/Regularizer_19/Square:y:0$kernel/Regularizer_19/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/Sum
kernel/Regularizer_19/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_19/mul/xจ
kernel/Regularizer_19/mulMul$kernel/Regularizer_19/mul/x:output:0"kernel/Regularizer_19/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_19/mulั
+kernel/Regularizer_20/Square/ReadVariableOpReadVariableOp-expand1x1fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_20/Square/ReadVariableOpญ
kernel/Regularizer_20/SquareSquare3kernel/Regularizer_20/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_20/Square
kernel/Regularizer_20/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_20/Constฆ
kernel/Regularizer_20/SumSum kernel/Regularizer_20/Square:y:0$kernel/Regularizer_20/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/Sum
kernel/Regularizer_20/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_20/mul/xจ
kernel/Regularizer_20/mulMul$kernel/Regularizer_20/mul/x:output:0"kernel/Regularizer_20/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_20/mulั
+kernel/Regularizer_21/Square/ReadVariableOpReadVariableOp-expand3x3fire8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_21/Square/ReadVariableOpญ
kernel/Regularizer_21/SquareSquare3kernel/Regularizer_21/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_21/Square
kernel/Regularizer_21/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_21/Constฆ
kernel/Regularizer_21/SumSum kernel/Regularizer_21/Square:y:0$kernel/Regularizer_21/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/Sum
kernel/Regularizer_21/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_21/mul/xจ
kernel/Regularizer_21/mulMul$kernel/Regularizer_21/mul/x:output:0"kernel/Regularizer_21/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_21/mulฯ
+kernel/Regularizer_22/Square/ReadVariableOpReadVariableOp+squeezefire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_22/Square/ReadVariableOpญ
kernel/Regularizer_22/SquareSquare3kernel/Regularizer_22/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_22/Square
kernel/Regularizer_22/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_22/Constฆ
kernel/Regularizer_22/SumSum kernel/Regularizer_22/Square:y:0$kernel/Regularizer_22/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/Sum
kernel/Regularizer_22/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_22/mul/xจ
kernel/Regularizer_22/mulMul$kernel/Regularizer_22/mul/x:output:0"kernel/Regularizer_22/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_22/mulั
+kernel/Regularizer_23/Square/ReadVariableOpReadVariableOp-expand1x1fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_23/Square/ReadVariableOpญ
kernel/Regularizer_23/SquareSquare3kernel/Regularizer_23/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_23/Square
kernel/Regularizer_23/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_23/Constฆ
kernel/Regularizer_23/SumSum kernel/Regularizer_23/Square:y:0$kernel/Regularizer_23/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/Sum
kernel/Regularizer_23/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_23/mul/xจ
kernel/Regularizer_23/mulMul$kernel/Regularizer_23/mul/x:output:0"kernel/Regularizer_23/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_23/mulั
+kernel/Regularizer_24/Square/ReadVariableOpReadVariableOp-expand3x3fire9_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+kernel/Regularizer_24/Square/ReadVariableOpญ
kernel/Regularizer_24/SquareSquare3kernel/Regularizer_24/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2
kernel/Regularizer_24/Square
kernel/Regularizer_24/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_24/Constฆ
kernel/Regularizer_24/SumSum kernel/Regularizer_24/Square:y:0$kernel/Regularizer_24/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/Sum
kernel/Regularizer_24/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer_24/mul/xจ
kernel/Regularizer_24/mulMul$kernel/Regularizer_24/mul/x:output:0"kernel/Regularizer_24/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_24/mulp
IdentityIdentityDenseFinal/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapes๐
ํ:?????????เเ:::::::::::::::::::::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:?????????เเ
 
_user_specified_nameinputs
บ
ฒ
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_268674

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpฃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????77@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????77@2
Reluป
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpฃ
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????77@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????77:::W S
/
_output_shapes
:?????????77
 
_user_specified_nameinputs
ภ	
g
__inference_loss_fn_18_2696495
1kernel_regularizer_square_readvariableop_resource
identityฯ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:0ภ*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:0ภ2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ย
ฒ
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_268892

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpค
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Reluผ
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype02*
(kernel/Regularizer/Square/ReadVariableOpค
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
: 2
kernel/Regularizer/Square
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ื#<2
kernel/Regularizer/mul/x
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
ม
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_265974

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? R 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:?????????ค2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:?????????ค2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


/__inference_Expand1x1Fire7_layer_call_fn_269119

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????ภ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_2656422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????ภ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs"ธL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ณ
serving_default
A
Input8
serving_default_Input:0?????????เเ>

DenseFinal0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:กช

ื
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer_with_weights-12
layer-18
layer-19
layer_with_weights-13
layer-20
layer_with_weights-14
layer-21
layer_with_weights-15
layer-22
layer-23
layer_with_weights-16
layer-24
layer_with_weights-17
layer-25
layer_with_weights-18
layer-26
layer-27
layer_with_weights-19
layer-28
layer_with_weights-20
layer-29
layer_with_weights-21
layer-30
 layer-31
!layer-32
"layer_with_weights-22
"layer-33
#layer_with_weights-23
#layer-34
$layer_with_weights-24
$layer-35
%layer-36
&layer-37
'layer-38
(layer_with_weights-25
(layer-39
#)_self_saveable_object_factories
*	optimizer
+
signatures
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?__call__
_default_save_signature
+&call_and_return_all_conditional_losses"
_tf_keras_network์{"class_name": "Functional", "name": "SqueezeNet_Preloaded", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "SqueezeNet_Preloaded", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "Conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv2D_1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "MaxPool1", "inbound_nodes": [[["Conv2D_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire2", "inbound_nodes": [[["MaxPool1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire2", "inbound_nodes": [[["SqueezeFire2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire2", "inbound_nodes": [[["SqueezeFire2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate2", "inbound_nodes": [[["Expand1x1Fire2", 0, 0, {}], ["Expand3x3Fire2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire3", "inbound_nodes": [[["Concatenate2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire3", "inbound_nodes": [[["SqueezeFire3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire3", "inbound_nodes": [[["SqueezeFire3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate3", "inbound_nodes": [[["Expand1x1Fire3", 0, 0, {}], ["Expand3x3Fire3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire4", "inbound_nodes": [[["Concatenate3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire4", "inbound_nodes": [[["SqueezeFire4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire4", "inbound_nodes": [[["SqueezeFire4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate4", "inbound_nodes": [[["Expand1x1Fire4", 0, 0, {}], ["Expand3x3Fire4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "MaxPool4", "inbound_nodes": [[["Concatenate4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire5", "inbound_nodes": [[["MaxPool4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire5", "inbound_nodes": [[["SqueezeFire5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire5", "inbound_nodes": [[["SqueezeFire5", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate5", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate5", "inbound_nodes": [[["Expand1x1Fire5", 0, 0, {}], ["Expand3x3Fire5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire6", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire6", "inbound_nodes": [[["Concatenate5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire6", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire6", "inbound_nodes": [[["SqueezeFire6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire6", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire6", "inbound_nodes": [[["SqueezeFire6", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate6", "inbound_nodes": [[["Expand1x1Fire6", 0, 0, {}], ["Expand3x3Fire6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire7", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire7", "inbound_nodes": [[["Concatenate6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire7", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire7", "inbound_nodes": [[["SqueezeFire7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire7", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire7", "inbound_nodes": [[["SqueezeFire7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate7", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate7", "inbound_nodes": [[["Expand1x1Fire7", 0, 0, {}], ["Expand3x3Fire7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire8", "inbound_nodes": [[["Concatenate7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire8", "inbound_nodes": [[["SqueezeFire8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire8", "inbound_nodes": [[["SqueezeFire8", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate8", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate8", "inbound_nodes": [[["Expand1x1Fire8", 0, 0, {}], ["Expand3x3Fire8", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "MaxPool8", "inbound_nodes": [[["Concatenate8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire9", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire9", "inbound_nodes": [[["MaxPool8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire9", "inbound_nodes": [[["SqueezeFire9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire9", "inbound_nodes": [[["SqueezeFire9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate9", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate9", "inbound_nodes": [[["Expand1x1Fire9", 0, 0, {}], ["Expand3x3Fire9", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "Dropout9", "inbound_nodes": [[["Concatenate9", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["Dropout9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "DenseFinal", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "DenseFinal", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["DenseFinal", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "SqueezeNet_Preloaded", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "Conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv2D_1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "MaxPool1", "inbound_nodes": [[["Conv2D_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire2", "inbound_nodes": [[["MaxPool1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire2", "inbound_nodes": [[["SqueezeFire2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire2", "inbound_nodes": [[["SqueezeFire2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate2", "inbound_nodes": [[["Expand1x1Fire2", 0, 0, {}], ["Expand3x3Fire2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire3", "inbound_nodes": [[["Concatenate2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire3", "inbound_nodes": [[["SqueezeFire3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire3", "inbound_nodes": [[["SqueezeFire3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate3", "inbound_nodes": [[["Expand1x1Fire3", 0, 0, {}], ["Expand3x3Fire3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire4", "inbound_nodes": [[["Concatenate3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire4", "inbound_nodes": [[["SqueezeFire4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire4", "inbound_nodes": [[["SqueezeFire4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate4", "inbound_nodes": [[["Expand1x1Fire4", 0, 0, {}], ["Expand3x3Fire4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "MaxPool4", "inbound_nodes": [[["Concatenate4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire5", "inbound_nodes": [[["MaxPool4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire5", "inbound_nodes": [[["SqueezeFire5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire5", "inbound_nodes": [[["SqueezeFire5", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate5", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate5", "inbound_nodes": [[["Expand1x1Fire5", 0, 0, {}], ["Expand3x3Fire5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire6", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire6", "inbound_nodes": [[["Concatenate5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire6", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire6", "inbound_nodes": [[["SqueezeFire6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire6", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire6", "inbound_nodes": [[["SqueezeFire6", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate6", "inbound_nodes": [[["Expand1x1Fire6", 0, 0, {}], ["Expand3x3Fire6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire7", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire7", "inbound_nodes": [[["Concatenate6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire7", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire7", "inbound_nodes": [[["SqueezeFire7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire7", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire7", "inbound_nodes": [[["SqueezeFire7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate7", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate7", "inbound_nodes": [[["Expand1x1Fire7", 0, 0, {}], ["Expand3x3Fire7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire8", "inbound_nodes": [[["Concatenate7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire8", "inbound_nodes": [[["SqueezeFire8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire8", "inbound_nodes": [[["SqueezeFire8", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate8", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate8", "inbound_nodes": [[["Expand1x1Fire8", 0, 0, {}], ["Expand3x3Fire8", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "MaxPool8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "MaxPool8", "inbound_nodes": [[["Concatenate8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "SqueezeFire9", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SqueezeFire9", "inbound_nodes": [[["MaxPool8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand1x1Fire9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand1x1Fire9", "inbound_nodes": [[["SqueezeFire9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Expand3x3Fire9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Expand3x3Fire9", "inbound_nodes": [[["SqueezeFire9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate9", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate9", "inbound_nodes": [[["Expand1x1Fire9", 0, 0, {}], ["Expand3x3Fire9", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "Dropout9", "inbound_nodes": [[["Concatenate9", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["Dropout9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "DenseFinal", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "DenseFinal", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["DenseFinal", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "clipnorm": 1, "learning_rate": 0.009999999776482582, "decay": 9.999999747378752e-05, "momentum": 0.0, "nesterov": false}}}}

#0_self_saveable_object_factories"๖
_tf_keras_input_layerึ{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
ั


1kernel
2bias
#3_self_saveable_object_factories
4regularization_losses
5	variables
6trainable_variables
7	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๋{"class_name": "Conv2D", "name": "Conv2D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}}

#8_self_saveable_object_factories
9regularization_losses
:	variables
;trainable_variables
<	keras_api
__call__
+&call_and_return_all_conditional_losses"โ
_tf_keras_layerศ{"class_name": "MaxPooling2D", "name": "MaxPool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MaxPool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ู


=kernel
>bias
#?_self_saveable_object_factories
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๓{"class_name": "Conv2D", "name": "SqueezeFire2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SqueezeFire2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 64]}}
?


Dkernel
Ebias
#F_self_saveable_object_factories
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๗{"class_name": "Conv2D", "name": "Expand1x1Fire2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand1x1Fire2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 16]}}
?


Kkernel
Lbias
#M_self_saveable_object_factories
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๗{"class_name": "Conv2D", "name": "Expand3x3Fire2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand3x3Fire2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 16]}}

#R_self_saveable_object_factories
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
__call__
+&call_and_return_all_conditional_losses"ฬ
_tf_keras_layerฒ{"class_name": "Concatenate", "name": "Concatenate2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 55, 55, 64]}, {"class_name": "TensorShape", "items": [null, 55, 55, 64]}]}
?


Wkernel
Xbias
#Y_self_saveable_object_factories
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๕{"class_name": "Conv2D", "name": "SqueezeFire3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SqueezeFire3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 128]}}
?


^kernel
_bias
#`_self_saveable_object_factories
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๗{"class_name": "Conv2D", "name": "Expand1x1Fire3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand1x1Fire3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 16]}}
?


ekernel
fbias
#g_self_saveable_object_factories
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๗{"class_name": "Conv2D", "name": "Expand3x3Fire3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand3x3Fire3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 16]}}

#l_self_saveable_object_factories
mregularization_losses
n	variables
otrainable_variables
p	keras_api
__call__
+&call_and_return_all_conditional_losses"ฬ
_tf_keras_layerฒ{"class_name": "Concatenate", "name": "Concatenate3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 55, 55, 64]}, {"class_name": "TensorShape", "items": [null, 55, 55, 64]}]}
?


qkernel
rbias
#s_self_saveable_object_factories
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๕{"class_name": "Conv2D", "name": "SqueezeFire4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SqueezeFire4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 128]}}
?


xkernel
ybias
#z_self_saveable_object_factories
{regularization_losses
|	variables
}trainable_variables
~	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand1x1Fire4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand1x1Fire4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 32]}}
ไ


kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand3x3Fire4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand3x3Fire4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 55, 32]}}

$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ฮ
_tf_keras_layerด{"class_name": "Concatenate", "name": "Concatenate4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate4", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 55, 55, 128]}, {"class_name": "TensorShape", "items": [null, 55, 55, 128]}]}

$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"โ
_tf_keras_layerศ{"class_name": "MaxPooling2D", "name": "MaxPool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MaxPool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
โ

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+ก&call_and_return_all_conditional_losses"	
_tf_keras_layer๕{"class_name": "Conv2D", "name": "SqueezeFire5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SqueezeFire5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 256]}}
ๅ

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
ข__call__
+ฃ&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand1x1Fire5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand1x1Fire5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 32]}}
ๅ

kernel
	bias
$?_self_saveable_object_factories
กregularization_losses
ข	variables
ฃtrainable_variables
ค	keras_api
ค__call__
+ฅ&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand3x3Fire5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand3x3Fire5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 32]}}

$ฅ_self_saveable_object_factories
ฆregularization_losses
ง	variables
จtrainable_variables
ฉ	keras_api
ฆ__call__
+ง&call_and_return_all_conditional_losses"ฮ
_tf_keras_layerด{"class_name": "Concatenate", "name": "Concatenate5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate5", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 27, 27, 128]}, {"class_name": "TensorShape", "items": [null, 27, 27, 128]}]}
โ

ชkernel
	ซbias
$ฌ_self_saveable_object_factories
ญregularization_losses
ฎ	variables
ฏtrainable_variables
ฐ	keras_api
จ__call__
+ฉ&call_and_return_all_conditional_losses"	
_tf_keras_layer๕{"class_name": "Conv2D", "name": "SqueezeFire6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SqueezeFire6", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 256]}}
ๅ

ฑkernel
	ฒbias
$ณ_self_saveable_object_factories
ดregularization_losses
ต	variables
ถtrainable_variables
ท	keras_api
ช__call__
+ซ&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand1x1Fire6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand1x1Fire6", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 48]}}
ๅ

ธkernel
	นbias
$บ_self_saveable_object_factories
ปregularization_losses
ผ	variables
ฝtrainable_variables
พ	keras_api
ฌ__call__
+ญ&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand3x3Fire6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand3x3Fire6", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 48]}}

$ฟ_self_saveable_object_factories
ภregularization_losses
ม	variables
ยtrainable_variables
ร	keras_api
ฎ__call__
+ฏ&call_and_return_all_conditional_losses"ฮ
_tf_keras_layerด{"class_name": "Concatenate", "name": "Concatenate6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate6", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 27, 27, 192]}, {"class_name": "TensorShape", "items": [null, 27, 27, 192]}]}
โ

ฤkernel
	ลbias
$ฦ_self_saveable_object_factories
วregularization_losses
ศ	variables
ษtrainable_variables
ส	keras_api
ฐ__call__
+ฑ&call_and_return_all_conditional_losses"	
_tf_keras_layer๕{"class_name": "Conv2D", "name": "SqueezeFire7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SqueezeFire7", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 384}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 384]}}
ๅ

หkernel
	ฬbias
$อ_self_saveable_object_factories
ฮregularization_losses
ฯ	variables
ะtrainable_variables
ั	keras_api
ฒ__call__
+ณ&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand1x1Fire7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand1x1Fire7", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 48]}}
ๅ

าkernel
	ำbias
$ิ_self_saveable_object_factories
ีregularization_losses
ึ	variables
ืtrainable_variables
ุ	keras_api
ด__call__
+ต&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand3x3Fire7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand3x3Fire7", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 48]}}

$ู_self_saveable_object_factories
ฺregularization_losses
?	variables
?trainable_variables
?	keras_api
ถ__call__
+ท&call_and_return_all_conditional_losses"ฮ
_tf_keras_layerด{"class_name": "Concatenate", "name": "Concatenate7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate7", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 27, 27, 192]}, {"class_name": "TensorShape", "items": [null, 27, 27, 192]}]}
โ

?kernel
	฿bias
$เ_self_saveable_object_factories
แregularization_losses
โ	variables
ใtrainable_variables
ไ	keras_api
ธ__call__
+น&call_and_return_all_conditional_losses"	
_tf_keras_layer๕{"class_name": "Conv2D", "name": "SqueezeFire8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SqueezeFire8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 384}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 384]}}
ๅ

ๅkernel
	ๆbias
$็_self_saveable_object_factories
่regularization_losses
้	variables
๊trainable_variables
๋	keras_api
บ__call__
+ป&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand1x1Fire8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand1x1Fire8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 64]}}
ๅ

์kernel
	ํbias
$๎_self_saveable_object_factories
๏regularization_losses
๐	variables
๑trainable_variables
๒	keras_api
ผ__call__
+ฝ&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand3x3Fire8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand3x3Fire8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 64]}}

$๓_self_saveable_object_factories
๔regularization_losses
๕	variables
๖trainable_variables
๗	keras_api
พ__call__
+ฟ&call_and_return_all_conditional_losses"ฮ
_tf_keras_layerด{"class_name": "Concatenate", "name": "Concatenate8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate8", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 27, 27, 256]}, {"class_name": "TensorShape", "items": [null, 27, 27, 256]}]}

$๘_self_saveable_object_factories
๙regularization_losses
๚	variables
๛trainable_variables
?	keras_api
ภ__call__
+ม&call_and_return_all_conditional_losses"โ
_tf_keras_layerศ{"class_name": "MaxPooling2D", "name": "MaxPool8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MaxPool8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
โ

?kernel
	?bias
$?_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
ย__call__
+ร&call_and_return_all_conditional_losses"	
_tf_keras_layer๕{"class_name": "Conv2D", "name": "SqueezeFire9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SqueezeFire9", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 512]}}
ๅ

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
ฤ__call__
+ล&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand1x1Fire9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand1x1Fire9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}
ๅ

kernel
	bias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
ฦ__call__
+ว&call_and_return_all_conditional_losses"	
_tf_keras_layer๘{"class_name": "Conv2D", "name": "Expand3x3Fire9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Expand3x3Fire9", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}

$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
ศ__call__
+ษ&call_and_return_all_conditional_losses"ฮ
_tf_keras_layerด{"class_name": "Concatenate", "name": "Concatenate9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate9", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 13, 13, 256]}, {"class_name": "TensorShape", "items": [null, 13, 13, 256]}]}

$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
ส__call__
+ห&call_and_return_all_conditional_losses"ิ
_tf_keras_layerบ{"class_name": "Dropout", "name": "Dropout9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dropout9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}

$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
?	keras_api
ฬ__call__
+อ&call_and_return_all_conditional_losses"ื
_tf_keras_layerฝ{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ฌ
กkernel
	ขbias
$ฃ_self_saveable_object_factories
คregularization_losses
ฅ	variables
ฆtrainable_variables
ง	keras_api
ฮ__call__
+ฯ&call_and_return_all_conditional_losses"ู
_tf_keras_layerฟ{"class_name": "Dense", "name": "DenseFinal", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "DenseFinal", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 86528}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 86528]}}
 "
trackable_dict_wrapper
M
	จiter

ฉdecay
ชlearning_rate
ซmomentum"
	optimizer
-
ะserving_default"
signature_map
๗
ั0
า1
ำ2
ิ3
ี4
ึ5
ื6
ุ7
ู8
ฺ9
?10
?11
?12
?13
฿14
เ15
แ16
โ17
ใ18
ไ19
ๅ20
ๆ21
็22
่23
้24"
trackable_list_wrapper
ื
10
21
=2
>3
D4
E5
K6
L7
W8
X9
^10
_11
e12
f13
q14
r15
x16
y17
18
19
20
21
22
23
24
25
ช26
ซ27
ฑ28
ฒ29
ธ30
น31
ฤ32
ล33
ห34
ฬ35
า36
ำ37
?38
฿39
ๅ40
ๆ41
์42
ํ43
?44
?45
46
47
48
49
ก50
ข51"
trackable_list_wrapper
ื
10
21
=2
>3
D4
E5
K6
L7
W8
X9
^10
_11
e12
f13
q14
r15
x16
y17
18
19
20
21
22
23
24
25
ช26
ซ27
ฑ28
ฒ29
ธ30
น31
ฤ32
ล33
ห34
ฬ35
า36
ำ37
?38
฿39
ๅ40
ๆ41
์42
ํ43
?44
?45
46
47
48
49
ก50
ข51"
trackable_list_wrapper
ำ
,regularization_losses
ฌnon_trainable_variables
ญlayers
ฎmetrics
-	variables
 ฏlayer_regularization_losses
.trainable_variables
ฐlayer_metrics
?__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
):'@2Conv2D_1/kernel
:@2Conv2D_1/bias
 "
trackable_dict_wrapper
(
ั0"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
ต
4regularization_losses
ฑnon_trainable_variables
ฒlayers
ณmetrics
5	variables
 ดlayer_regularization_losses
6trainable_variables
ตlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
9regularization_losses
ถnon_trainable_variables
ทlayers
ธmetrics
:	variables
 นlayer_regularization_losses
;trainable_variables
บlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-:+@2SqueezeFire2/kernel
:2SqueezeFire2/bias
 "
trackable_dict_wrapper
(
า0"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
ต
@regularization_losses
ปnon_trainable_variables
ผlayers
ฝmetrics
A	variables
 พlayer_regularization_losses
Btrainable_variables
ฟlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-@2Expand1x1Fire2/kernel
!:@2Expand1x1Fire2/bias
 "
trackable_dict_wrapper
(
ำ0"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
ต
Gregularization_losses
ภnon_trainable_variables
มlayers
ยmetrics
H	variables
 รlayer_regularization_losses
Itrainable_variables
ฤlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-@2Expand3x3Fire2/kernel
!:@2Expand3x3Fire2/bias
 "
trackable_dict_wrapper
(
ิ0"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
ต
Nregularization_losses
ลnon_trainable_variables
ฦlayers
วmetrics
O	variables
 ศlayer_regularization_losses
Ptrainable_variables
ษlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
Sregularization_losses
สnon_trainable_variables
หlayers
ฬmetrics
T	variables
 อlayer_regularization_losses
Utrainable_variables
ฮlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,2SqueezeFire3/kernel
:2SqueezeFire3/bias
 "
trackable_dict_wrapper
(
ี0"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
ต
Zregularization_losses
ฯnon_trainable_variables
ะlayers
ัmetrics
[	variables
 าlayer_regularization_losses
\trainable_variables
ำlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-@2Expand1x1Fire3/kernel
!:@2Expand1x1Fire3/bias
 "
trackable_dict_wrapper
(
ึ0"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
ต
aregularization_losses
ิnon_trainable_variables
ีlayers
ึmetrics
b	variables
 ืlayer_regularization_losses
ctrainable_variables
ุlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-@2Expand3x3Fire3/kernel
!:@2Expand3x3Fire3/bias
 "
trackable_dict_wrapper
(
ื0"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
ต
hregularization_losses
ูnon_trainable_variables
ฺlayers
?metrics
i	variables
 ?layer_regularization_losses
jtrainable_variables
?layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
mregularization_losses
?non_trainable_variables
฿layers
เmetrics
n	variables
 แlayer_regularization_losses
otrainable_variables
โlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:, 2SqueezeFire4/kernel
: 2SqueezeFire4/bias
 "
trackable_dict_wrapper
(
ุ0"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
ต
tregularization_losses
ใnon_trainable_variables
ไlayers
ๅmetrics
u	variables
 ๆlayer_regularization_losses
vtrainable_variables
็layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0:. 2Expand1x1Fire4/kernel
": 2Expand1x1Fire4/bias
 "
trackable_dict_wrapper
(
ู0"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
ต
{regularization_losses
่non_trainable_variables
้layers
๊metrics
|	variables
 ๋layer_regularization_losses
}trainable_variables
์layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0:. 2Expand3x3Fire4/kernel
": 2Expand3x3Fire4/bias
 "
trackable_dict_wrapper
(
ฺ0"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
ธ
regularization_losses
ํnon_trainable_variables
๎layers
๏metrics
	variables
 ๐layer_regularization_losses
trainable_variables
๑layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
๒non_trainable_variables
๓layers
๔metrics
	variables
 ๕layer_regularization_losses
trainable_variables
๖layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
๗non_trainable_variables
๘layers
๙metrics
	variables
 ๚layer_regularization_losses
trainable_variables
๛layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:, 2SqueezeFire5/kernel
: 2SqueezeFire5/bias
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
ธ
regularization_losses
?non_trainable_variables
?layers
?metrics
	variables
 ?layer_regularization_losses
trainable_variables
layer_metrics
?__call__
+ก&call_and_return_all_conditional_losses
'ก"call_and_return_conditional_losses"
_generic_user_object
0:. 2Expand1x1Fire5/kernel
": 2Expand1x1Fire5/bias
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
ธ
regularization_losses
non_trainable_variables
layers
metrics
	variables
 layer_regularization_losses
trainable_variables
layer_metrics
ข__call__
+ฃ&call_and_return_all_conditional_losses
'ฃ"call_and_return_conditional_losses"
_generic_user_object
0:. 2Expand3x3Fire5/kernel
": 2Expand3x3Fire5/bias
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
ธ
กregularization_losses
non_trainable_variables
layers
metrics
ข	variables
 layer_regularization_losses
ฃtrainable_variables
layer_metrics
ค__call__
+ฅ&call_and_return_all_conditional_losses
'ฅ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ฆregularization_losses
non_trainable_variables
layers
metrics
ง	variables
 layer_regularization_losses
จtrainable_variables
layer_metrics
ฆ__call__
+ง&call_and_return_all_conditional_losses
'ง"call_and_return_conditional_losses"
_generic_user_object
.:,02SqueezeFire6/kernel
:02SqueezeFire6/bias
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
0
ช0
ซ1"
trackable_list_wrapper
0
ช0
ซ1"
trackable_list_wrapper
ธ
ญregularization_losses
non_trainable_variables
layers
metrics
ฎ	variables
 layer_regularization_losses
ฏtrainable_variables
layer_metrics
จ__call__
+ฉ&call_and_return_all_conditional_losses
'ฉ"call_and_return_conditional_losses"
_generic_user_object
0:.0ภ2Expand1x1Fire6/kernel
": ภ2Expand1x1Fire6/bias
 "
trackable_dict_wrapper
(
฿0"
trackable_list_wrapper
0
ฑ0
ฒ1"
trackable_list_wrapper
0
ฑ0
ฒ1"
trackable_list_wrapper
ธ
ดregularization_losses
non_trainable_variables
layers
metrics
ต	variables
 layer_regularization_losses
ถtrainable_variables
layer_metrics
ช__call__
+ซ&call_and_return_all_conditional_losses
'ซ"call_and_return_conditional_losses"
_generic_user_object
0:.0ภ2Expand3x3Fire6/kernel
": ภ2Expand3x3Fire6/bias
 "
trackable_dict_wrapper
(
เ0"
trackable_list_wrapper
0
ธ0
น1"
trackable_list_wrapper
0
ธ0
น1"
trackable_list_wrapper
ธ
ปregularization_losses
non_trainable_variables
layers
metrics
ผ	variables
 layer_regularization_losses
ฝtrainable_variables
layer_metrics
ฌ__call__
+ญ&call_and_return_all_conditional_losses
'ญ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ภregularization_losses
non_trainable_variables
?layers
กmetrics
ม	variables
 ขlayer_regularization_losses
ยtrainable_variables
ฃlayer_metrics
ฎ__call__
+ฏ&call_and_return_all_conditional_losses
'ฏ"call_and_return_conditional_losses"
_generic_user_object
.:,02SqueezeFire7/kernel
:02SqueezeFire7/bias
 "
trackable_dict_wrapper
(
แ0"
trackable_list_wrapper
0
ฤ0
ล1"
trackable_list_wrapper
0
ฤ0
ล1"
trackable_list_wrapper
ธ
วregularization_losses
คnon_trainable_variables
ฅlayers
ฆmetrics
ศ	variables
 งlayer_regularization_losses
ษtrainable_variables
จlayer_metrics
ฐ__call__
+ฑ&call_and_return_all_conditional_losses
'ฑ"call_and_return_conditional_losses"
_generic_user_object
0:.0ภ2Expand1x1Fire7/kernel
": ภ2Expand1x1Fire7/bias
 "
trackable_dict_wrapper
(
โ0"
trackable_list_wrapper
0
ห0
ฬ1"
trackable_list_wrapper
0
ห0
ฬ1"
trackable_list_wrapper
ธ
ฮregularization_losses
ฉnon_trainable_variables
ชlayers
ซmetrics
ฯ	variables
 ฌlayer_regularization_losses
ะtrainable_variables
ญlayer_metrics
ฒ__call__
+ณ&call_and_return_all_conditional_losses
'ณ"call_and_return_conditional_losses"
_generic_user_object
0:.0ภ2Expand3x3Fire7/kernel
": ภ2Expand3x3Fire7/bias
 "
trackable_dict_wrapper
(
ใ0"
trackable_list_wrapper
0
า0
ำ1"
trackable_list_wrapper
0
า0
ำ1"
trackable_list_wrapper
ธ
ีregularization_losses
ฎnon_trainable_variables
ฏlayers
ฐmetrics
ึ	variables
 ฑlayer_regularization_losses
ืtrainable_variables
ฒlayer_metrics
ด__call__
+ต&call_and_return_all_conditional_losses
'ต"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ฺregularization_losses
ณnon_trainable_variables
ดlayers
ตmetrics
?	variables
 ถlayer_regularization_losses
?trainable_variables
ทlayer_metrics
ถ__call__
+ท&call_and_return_all_conditional_losses
'ท"call_and_return_conditional_losses"
_generic_user_object
.:,@2SqueezeFire8/kernel
:@2SqueezeFire8/bias
 "
trackable_dict_wrapper
(
ไ0"
trackable_list_wrapper
0
?0
฿1"
trackable_list_wrapper
0
?0
฿1"
trackable_list_wrapper
ธ
แregularization_losses
ธnon_trainable_variables
นlayers
บmetrics
โ	variables
 ปlayer_regularization_losses
ใtrainable_variables
ผlayer_metrics
ธ__call__
+น&call_and_return_all_conditional_losses
'น"call_and_return_conditional_losses"
_generic_user_object
0:.@2Expand1x1Fire8/kernel
": 2Expand1x1Fire8/bias
 "
trackable_dict_wrapper
(
ๅ0"
trackable_list_wrapper
0
ๅ0
ๆ1"
trackable_list_wrapper
0
ๅ0
ๆ1"
trackable_list_wrapper
ธ
่regularization_losses
ฝnon_trainable_variables
พlayers
ฟmetrics
้	variables
 ภlayer_regularization_losses
๊trainable_variables
มlayer_metrics
บ__call__
+ป&call_and_return_all_conditional_losses
'ป"call_and_return_conditional_losses"
_generic_user_object
0:.@2Expand3x3Fire8/kernel
": 2Expand3x3Fire8/bias
 "
trackable_dict_wrapper
(
ๆ0"
trackable_list_wrapper
0
์0
ํ1"
trackable_list_wrapper
0
์0
ํ1"
trackable_list_wrapper
ธ
๏regularization_losses
ยnon_trainable_variables
รlayers
ฤmetrics
๐	variables
 ลlayer_regularization_losses
๑trainable_variables
ฦlayer_metrics
ผ__call__
+ฝ&call_and_return_all_conditional_losses
'ฝ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
๔regularization_losses
วnon_trainable_variables
ศlayers
ษmetrics
๕	variables
 สlayer_regularization_losses
๖trainable_variables
หlayer_metrics
พ__call__
+ฟ&call_and_return_all_conditional_losses
'ฟ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
๙regularization_losses
ฬnon_trainable_variables
อlayers
ฮmetrics
๚	variables
 ฯlayer_regularization_losses
๛trainable_variables
ะlayer_metrics
ภ__call__
+ม&call_and_return_all_conditional_losses
'ม"call_and_return_conditional_losses"
_generic_user_object
.:,@2SqueezeFire9/kernel
:@2SqueezeFire9/bias
 "
trackable_dict_wrapper
(
็0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
ธ
regularization_losses
ัnon_trainable_variables
าlayers
ำmetrics
	variables
 ิlayer_regularization_losses
trainable_variables
ีlayer_metrics
ย__call__
+ร&call_and_return_all_conditional_losses
'ร"call_and_return_conditional_losses"
_generic_user_object
0:.@2Expand1x1Fire9/kernel
": 2Expand1x1Fire9/bias
 "
trackable_dict_wrapper
(
่0"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
ธ
regularization_losses
ึnon_trainable_variables
ืlayers
ุmetrics
	variables
 ูlayer_regularization_losses
trainable_variables
ฺlayer_metrics
ฤ__call__
+ล&call_and_return_all_conditional_losses
'ล"call_and_return_conditional_losses"
_generic_user_object
0:.@2Expand3x3Fire9/kernel
": 2Expand3x3Fire9/bias
 "
trackable_dict_wrapper
(
้0"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
ธ
regularization_losses
?non_trainable_variables
?layers
?metrics
	variables
 ?layer_regularization_losses
trainable_variables
฿layer_metrics
ฦ__call__
+ว&call_and_return_all_conditional_losses
'ว"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
เnon_trainable_variables
แlayers
โmetrics
	variables
 ใlayer_regularization_losses
trainable_variables
ไlayer_metrics
ศ__call__
+ษ&call_and_return_all_conditional_losses
'ษ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
ๅnon_trainable_variables
ๆlayers
็metrics
	variables
 ่layer_regularization_losses
trainable_variables
้layer_metrics
ส__call__
+ห&call_and_return_all_conditional_losses
'ห"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
๊non_trainable_variables
๋layers
์metrics
	variables
 ํlayer_regularization_losses
trainable_variables
๎layer_metrics
ฬ__call__
+อ&call_and_return_all_conditional_losses
'อ"call_and_return_conditional_losses"
_generic_user_object
%:#
ค2DenseFinal/kernel
:2DenseFinal/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
ก0
ข1"
trackable_list_wrapper
0
ก0
ข1"
trackable_list_wrapper
ธ
คregularization_losses
๏non_trainable_variables
๐layers
๑metrics
ฅ	variables
 ๒layer_regularization_losses
ฆtrainable_variables
๓layer_metrics
ฮ__call__
+ฯ&call_and_return_all_conditional_losses
'ฯ"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
ึ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39"
trackable_list_wrapper
0
๔0
๕1"
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
(
ั0"
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
(
า0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ำ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ิ0"
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
(
ี0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ึ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ื0"
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
(
ุ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ู0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ฺ0"
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
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
฿0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
เ0"
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
(
แ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
โ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ใ0"
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
(
ไ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ๅ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ๆ0"
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
(
็0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
่0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
้0"
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
ฟ

๖total

๗count
๘	variables
๙	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


๚total

๛count
?
_fn_kwargs
?	variables
?	keras_api"ธ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
๖0
๗1"
trackable_list_wrapper
.
๘	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
๚0
๛1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
ข2
5__inference_SqueezeNet_Preloaded_layer_call_fn_268369
5__inference_SqueezeNet_Preloaded_layer_call_fn_266864
5__inference_SqueezeNet_Preloaded_layer_call_fn_268478
5__inference_SqueezeNet_Preloaded_layer_call_fn_267270ภ
ทฒณ
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
kwonlydefaultsช 
annotationsช *
 
็2ไ
!__inference__wrapped_model_264942พ
ฒ
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *.ข+
)&
Input?????????เเ
2
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_266457
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_267902
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_266160
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_268260ภ
ทฒณ
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
kwonlydefaultsช 
annotationsช *
 
ำ2ะ
)__inference_Conv2D_1_layer_call_fn_268510ข
ฒ
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
annotationsช *
 
๎2๋
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_268501ข
ฒ
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
annotationsช *
 
2
)__inference_MaxPool1_layer_call_fn_264954เ
ฒ
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
annotationsช *@ข=
;84????????????????????????????????????
ฌ2ฉ
D__inference_MaxPool1_layer_call_and_return_conditional_losses_264948เ
ฒ
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
annotationsช *@ข=
;84????????????????????????????????????
ื2ิ
-__inference_SqueezeFire2_layer_call_fn_268542ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_268533ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand1x1Fire2_layer_call_fn_268574ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_268565ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand3x3Fire2_layer_call_fn_268606ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_268597ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_Concatenate2_layer_call_fn_268619ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_Concatenate2_layer_call_and_return_conditional_losses_268613ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_SqueezeFire3_layer_call_fn_268651ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_268642ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand1x1Fire3_layer_call_fn_268683ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_268674ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand3x3Fire3_layer_call_fn_268715ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_268706ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_Concatenate3_layer_call_fn_268728ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_Concatenate3_layer_call_and_return_conditional_losses_268722ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_SqueezeFire4_layer_call_fn_268760ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_268751ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand1x1Fire4_layer_call_fn_268792ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_268783ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand3x3Fire4_layer_call_fn_268824ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_268815ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_Concatenate4_layer_call_fn_268837ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_Concatenate4_layer_call_and_return_conditional_losses_268831ข
ฒ
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
annotationsช *
 
2
)__inference_MaxPool4_layer_call_fn_264966เ
ฒ
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
annotationsช *@ข=
;84????????????????????????????????????
ฌ2ฉ
D__inference_MaxPool4_layer_call_and_return_conditional_losses_264960เ
ฒ
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
annotationsช *@ข=
;84????????????????????????????????????
ื2ิ
-__inference_SqueezeFire5_layer_call_fn_268869ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_268860ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand1x1Fire5_layer_call_fn_268901ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_268892ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand3x3Fire5_layer_call_fn_268933ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_268924ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_Concatenate5_layer_call_fn_268946ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_Concatenate5_layer_call_and_return_conditional_losses_268940ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_SqueezeFire6_layer_call_fn_268978ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_268969ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand1x1Fire6_layer_call_fn_269010ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_269001ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand3x3Fire6_layer_call_fn_269042ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_269033ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_Concatenate6_layer_call_fn_269055ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_Concatenate6_layer_call_and_return_conditional_losses_269049ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_SqueezeFire7_layer_call_fn_269087ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_269078ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand1x1Fire7_layer_call_fn_269119ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_269110ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand3x3Fire7_layer_call_fn_269151ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_269142ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_Concatenate7_layer_call_fn_269164ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_Concatenate7_layer_call_and_return_conditional_losses_269158ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_SqueezeFire8_layer_call_fn_269196ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_269187ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand1x1Fire8_layer_call_fn_269228ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_269219ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand3x3Fire8_layer_call_fn_269260ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_269251ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_Concatenate8_layer_call_fn_269273ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_Concatenate8_layer_call_and_return_conditional_losses_269267ข
ฒ
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
annotationsช *
 
2
)__inference_MaxPool8_layer_call_fn_264978เ
ฒ
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
annotationsช *@ข=
;84????????????????????????????????????
ฌ2ฉ
D__inference_MaxPool8_layer_call_and_return_conditional_losses_264972เ
ฒ
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
annotationsช *@ข=
;84????????????????????????????????????
ื2ิ
-__inference_SqueezeFire9_layer_call_fn_269305ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_269296ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand1x1Fire9_layer_call_fn_269337ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_269328ข
ฒ
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
annotationsช *
 
ู2ึ
/__inference_Expand3x3Fire9_layer_call_fn_269369ข
ฒ
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
annotationsช *
 
๔2๑
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_269360ข
ฒ
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
annotationsช *
 
ื2ิ
-__inference_Concatenate9_layer_call_fn_269382ข
ฒ
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
annotationsช *
 
๒2๏
H__inference_Concatenate9_layer_call_and_return_conditional_losses_269376ข
ฒ
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
annotationsช *
 
2
)__inference_Dropout9_layer_call_fn_269404
)__inference_Dropout9_layer_call_fn_269409ด
ซฒง
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
kwonlydefaultsช 
annotationsช *
 
ฦ2ร
D__inference_Dropout9_layer_call_and_return_conditional_losses_269394
D__inference_Dropout9_layer_call_and_return_conditional_losses_269399ด
ซฒง
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
kwonlydefaultsช 
annotationsช *
 
ิ2ั
*__inference_flatten_1_layer_call_fn_269420ข
ฒ
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
annotationsช *
 
๏2์
E__inference_flatten_1_layer_call_and_return_conditional_losses_269415ข
ฒ
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
annotationsช *
 
ี2า
+__inference_DenseFinal_layer_call_fn_269440ข
ฒ
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
annotationsช *
 
๐2ํ
F__inference_DenseFinal_layer_call_and_return_conditional_losses_269431ข
ฒ
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
annotationsช *
 
1B/
$__inference_signature_wrapper_267537Input
ณ2ฐ
__inference_loss_fn_0_269451
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_1_269462
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_2_269473
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_3_269484
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_4_269495
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_5_269506
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_6_269517
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_7_269528
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_8_269539
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ณ2ฐ
__inference_loss_fn_9_269550
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_10_269561
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_11_269572
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_12_269583
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_13_269594
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_14_269605
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_15_269616
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_16_269627
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_17_269638
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_18_269649
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_19_269660
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_20_269671
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_21_269682
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_22_269693
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_23_269704
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข 
ด2ฑ
__inference_loss_fn_24_269715
ฒ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *ข ้
H__inference_Concatenate2_layer_call_and_return_conditional_losses_268613jขg
`ข]
[X
*'
inputs/0?????????77@
*'
inputs/1?????????77@
ช ".ข+
$!
0?????????77
 ม
-__inference_Concatenate2_layer_call_fn_268619jขg
`ข]
[X
*'
inputs/0?????????77@
*'
inputs/1?????????77@
ช "!?????????77้
H__inference_Concatenate3_layer_call_and_return_conditional_losses_268722jขg
`ข]
[X
*'
inputs/0?????????77@
*'
inputs/1?????????77@
ช ".ข+
$!
0?????????77
 ม
-__inference_Concatenate3_layer_call_fn_268728jขg
`ข]
[X
*'
inputs/0?????????77@
*'
inputs/1?????????77@
ช "!?????????77๋
H__inference_Concatenate4_layer_call_and_return_conditional_losses_268831lขi
bข_
]Z
+(
inputs/0?????????77
+(
inputs/1?????????77
ช ".ข+
$!
0?????????77
 ร
-__inference_Concatenate4_layer_call_fn_268837lขi
bข_
]Z
+(
inputs/0?????????77
+(
inputs/1?????????77
ช "!?????????77๋
H__inference_Concatenate5_layer_call_and_return_conditional_losses_268940lขi
bข_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ช ".ข+
$!
0?????????
 ร
-__inference_Concatenate5_layer_call_fn_268946lขi
bข_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ช "!?????????๋
H__inference_Concatenate6_layer_call_and_return_conditional_losses_269049lขi
bข_
]Z
+(
inputs/0?????????ภ
+(
inputs/1?????????ภ
ช ".ข+
$!
0?????????
 ร
-__inference_Concatenate6_layer_call_fn_269055lขi
bข_
]Z
+(
inputs/0?????????ภ
+(
inputs/1?????????ภ
ช "!?????????๋
H__inference_Concatenate7_layer_call_and_return_conditional_losses_269158lขi
bข_
]Z
+(
inputs/0?????????ภ
+(
inputs/1?????????ภ
ช ".ข+
$!
0?????????
 ร
-__inference_Concatenate7_layer_call_fn_269164lขi
bข_
]Z
+(
inputs/0?????????ภ
+(
inputs/1?????????ภ
ช "!?????????๋
H__inference_Concatenate8_layer_call_and_return_conditional_losses_269267lขi
bข_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ช ".ข+
$!
0?????????
 ร
-__inference_Concatenate8_layer_call_fn_269273lขi
bข_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ช "!?????????๋
H__inference_Concatenate9_layer_call_and_return_conditional_losses_269376lขi
bข_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ช ".ข+
$!
0?????????
 ร
-__inference_Concatenate9_layer_call_fn_269382lขi
bข_
]Z
+(
inputs/0?????????
+(
inputs/1?????????
ช "!?????????ถ
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_268501n129ข6
/ข,
*'
inputs?????????เเ
ช "-ข*
# 
0?????????pp@
 
)__inference_Conv2D_1_layer_call_fn_268510a129ข6
/ข,
*'
inputs?????????เเ
ช " ?????????pp@ช
F__inference_DenseFinal_layer_call_and_return_conditional_losses_269431`กข1ข.
'ข$
"
inputs?????????ค
ช "%ข"

0?????????
 
+__inference_DenseFinal_layer_call_fn_269440Sกข1ข.
'ข$
"
inputs?????????ค
ช "?????????ถ
D__inference_Dropout9_layer_call_and_return_conditional_losses_269394n<ข9
2ข/
)&
inputs?????????
p
ช ".ข+
$!
0?????????
 ถ
D__inference_Dropout9_layer_call_and_return_conditional_losses_269399n<ข9
2ข/
)&
inputs?????????
p 
ช ".ข+
$!
0?????????
 
)__inference_Dropout9_layer_call_fn_269404a<ข9
2ข/
)&
inputs?????????
p
ช "!?????????
)__inference_Dropout9_layer_call_fn_269409a<ข9
2ข/
)&
inputs?????????
p 
ช "!?????????บ
J__inference_Expand1x1Fire2_layer_call_and_return_conditional_losses_268565lDE7ข4
-ข*
(%
inputs?????????77
ช "-ข*
# 
0?????????77@
 
/__inference_Expand1x1Fire2_layer_call_fn_268574_DE7ข4
-ข*
(%
inputs?????????77
ช " ?????????77@บ
J__inference_Expand1x1Fire3_layer_call_and_return_conditional_losses_268674l^_7ข4
-ข*
(%
inputs?????????77
ช "-ข*
# 
0?????????77@
 
/__inference_Expand1x1Fire3_layer_call_fn_268683_^_7ข4
-ข*
(%
inputs?????????77
ช " ?????????77@ป
J__inference_Expand1x1Fire4_layer_call_and_return_conditional_losses_268783mxy7ข4
-ข*
(%
inputs?????????77 
ช ".ข+
$!
0?????????77
 
/__inference_Expand1x1Fire4_layer_call_fn_268792`xy7ข4
-ข*
(%
inputs?????????77 
ช "!?????????77ฝ
J__inference_Expand1x1Fire5_layer_call_and_return_conditional_losses_268892o7ข4
-ข*
(%
inputs????????? 
ช ".ข+
$!
0?????????
 
/__inference_Expand1x1Fire5_layer_call_fn_268901b7ข4
-ข*
(%
inputs????????? 
ช "!?????????ฝ
J__inference_Expand1x1Fire6_layer_call_and_return_conditional_losses_269001oฑฒ7ข4
-ข*
(%
inputs?????????0
ช ".ข+
$!
0?????????ภ
 
/__inference_Expand1x1Fire6_layer_call_fn_269010bฑฒ7ข4
-ข*
(%
inputs?????????0
ช "!?????????ภฝ
J__inference_Expand1x1Fire7_layer_call_and_return_conditional_losses_269110oหฬ7ข4
-ข*
(%
inputs?????????0
ช ".ข+
$!
0?????????ภ
 
/__inference_Expand1x1Fire7_layer_call_fn_269119bหฬ7ข4
-ข*
(%
inputs?????????0
ช "!?????????ภฝ
J__inference_Expand1x1Fire8_layer_call_and_return_conditional_losses_269219oๅๆ7ข4
-ข*
(%
inputs?????????@
ช ".ข+
$!
0?????????
 
/__inference_Expand1x1Fire8_layer_call_fn_269228bๅๆ7ข4
-ข*
(%
inputs?????????@
ช "!?????????ฝ
J__inference_Expand1x1Fire9_layer_call_and_return_conditional_losses_269328o7ข4
-ข*
(%
inputs?????????@
ช ".ข+
$!
0?????????
 
/__inference_Expand1x1Fire9_layer_call_fn_269337b7ข4
-ข*
(%
inputs?????????@
ช "!?????????บ
J__inference_Expand3x3Fire2_layer_call_and_return_conditional_losses_268597lKL7ข4
-ข*
(%
inputs?????????77
ช "-ข*
# 
0?????????77@
 
/__inference_Expand3x3Fire2_layer_call_fn_268606_KL7ข4
-ข*
(%
inputs?????????77
ช " ?????????77@บ
J__inference_Expand3x3Fire3_layer_call_and_return_conditional_losses_268706lef7ข4
-ข*
(%
inputs?????????77
ช "-ข*
# 
0?????????77@
 
/__inference_Expand3x3Fire3_layer_call_fn_268715_ef7ข4
-ข*
(%
inputs?????????77
ช " ?????????77@ผ
J__inference_Expand3x3Fire4_layer_call_and_return_conditional_losses_268815n7ข4
-ข*
(%
inputs?????????77 
ช ".ข+
$!
0?????????77
 
/__inference_Expand3x3Fire4_layer_call_fn_268824a7ข4
-ข*
(%
inputs?????????77 
ช "!?????????77ฝ
J__inference_Expand3x3Fire5_layer_call_and_return_conditional_losses_268924o7ข4
-ข*
(%
inputs????????? 
ช ".ข+
$!
0?????????
 
/__inference_Expand3x3Fire5_layer_call_fn_268933b7ข4
-ข*
(%
inputs????????? 
ช "!?????????ฝ
J__inference_Expand3x3Fire6_layer_call_and_return_conditional_losses_269033oธน7ข4
-ข*
(%
inputs?????????0
ช ".ข+
$!
0?????????ภ
 
/__inference_Expand3x3Fire6_layer_call_fn_269042bธน7ข4
-ข*
(%
inputs?????????0
ช "!?????????ภฝ
J__inference_Expand3x3Fire7_layer_call_and_return_conditional_losses_269142oาำ7ข4
-ข*
(%
inputs?????????0
ช ".ข+
$!
0?????????ภ
 
/__inference_Expand3x3Fire7_layer_call_fn_269151bาำ7ข4
-ข*
(%
inputs?????????0
ช "!?????????ภฝ
J__inference_Expand3x3Fire8_layer_call_and_return_conditional_losses_269251o์ํ7ข4
-ข*
(%
inputs?????????@
ช ".ข+
$!
0?????????
 
/__inference_Expand3x3Fire8_layer_call_fn_269260b์ํ7ข4
-ข*
(%
inputs?????????@
ช "!?????????ฝ
J__inference_Expand3x3Fire9_layer_call_and_return_conditional_losses_269360o7ข4
-ข*
(%
inputs?????????@
ช ".ข+
$!
0?????????
 
/__inference_Expand3x3Fire9_layer_call_fn_269369b7ข4
-ข*
(%
inputs?????????@
ช "!?????????็
D__inference_MaxPool1_layer_call_and_return_conditional_losses_264948RขO
HขE
C@
inputs4????????????????????????????????????
ช "HขE
>;
04????????????????????????????????????
 ฟ
)__inference_MaxPool1_layer_call_fn_264954RขO
HขE
C@
inputs4????????????????????????????????????
ช ";84????????????????????????????????????็
D__inference_MaxPool4_layer_call_and_return_conditional_losses_264960RขO
HขE
C@
inputs4????????????????????????????????????
ช "HขE
>;
04????????????????????????????????????
 ฟ
)__inference_MaxPool4_layer_call_fn_264966RขO
HขE
C@
inputs4????????????????????????????????????
ช ";84????????????????????????????????????็
D__inference_MaxPool8_layer_call_and_return_conditional_losses_264972RขO
HขE
C@
inputs4????????????????????????????????????
ช "HขE
>;
04????????????????????????????????????
 ฟ
)__inference_MaxPool8_layer_call_fn_264978RขO
HขE
C@
inputs4????????????????????????????????????
ช ";84????????????????????????????????????ธ
H__inference_SqueezeFire2_layer_call_and_return_conditional_losses_268533l=>7ข4
-ข*
(%
inputs?????????77@
ช "-ข*
# 
0?????????77
 
-__inference_SqueezeFire2_layer_call_fn_268542_=>7ข4
-ข*
(%
inputs?????????77@
ช " ?????????77น
H__inference_SqueezeFire3_layer_call_and_return_conditional_losses_268642mWX8ข5
.ข+
)&
inputs?????????77
ช "-ข*
# 
0?????????77
 
-__inference_SqueezeFire3_layer_call_fn_268651`WX8ข5
.ข+
)&
inputs?????????77
ช " ?????????77น
H__inference_SqueezeFire4_layer_call_and_return_conditional_losses_268751mqr8ข5
.ข+
)&
inputs?????????77
ช "-ข*
# 
0?????????77 
 
-__inference_SqueezeFire4_layer_call_fn_268760`qr8ข5
.ข+
)&
inputs?????????77
ช " ?????????77 ป
H__inference_SqueezeFire5_layer_call_and_return_conditional_losses_268860o8ข5
.ข+
)&
inputs?????????
ช "-ข*
# 
0????????? 
 
-__inference_SqueezeFire5_layer_call_fn_268869b8ข5
.ข+
)&
inputs?????????
ช " ????????? ป
H__inference_SqueezeFire6_layer_call_and_return_conditional_losses_268969oชซ8ข5
.ข+
)&
inputs?????????
ช "-ข*
# 
0?????????0
 
-__inference_SqueezeFire6_layer_call_fn_268978bชซ8ข5
.ข+
)&
inputs?????????
ช " ?????????0ป
H__inference_SqueezeFire7_layer_call_and_return_conditional_losses_269078oฤล8ข5
.ข+
)&
inputs?????????
ช "-ข*
# 
0?????????0
 
-__inference_SqueezeFire7_layer_call_fn_269087bฤล8ข5
.ข+
)&
inputs?????????
ช " ?????????0ป
H__inference_SqueezeFire8_layer_call_and_return_conditional_losses_269187o?฿8ข5
.ข+
)&
inputs?????????
ช "-ข*
# 
0?????????@
 
-__inference_SqueezeFire8_layer_call_fn_269196b?฿8ข5
.ข+
)&
inputs?????????
ช " ?????????@ป
H__inference_SqueezeFire9_layer_call_and_return_conditional_losses_269296o??8ข5
.ข+
)&
inputs?????????
ช "-ข*
# 
0?????????@
 
-__inference_SqueezeFire9_layer_call_fn_269305b??8ข5
.ข+
)&
inputs?????????
ช " ?????????@
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_266160ภU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กข@ข=
6ข3
)&
Input?????????เเ
p

 
ช "%ข"

0?????????
 
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_266457ภU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กข@ข=
6ข3
)&
Input?????????เเ
p 

 
ช "%ข"

0?????????
 
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_267902มU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กขAข>
7ข4
*'
inputs?????????เเ
p

 
ช "%ข"

0?????????
 
P__inference_SqueezeNet_Preloaded_layer_call_and_return_conditional_losses_268260มU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กขAข>
7ข4
*'
inputs?????????เเ
p 

 
ช "%ข"

0?????????
 ํ
5__inference_SqueezeNet_Preloaded_layer_call_fn_266864ณU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กข@ข=
6ข3
)&
Input?????????เเ
p

 
ช "?????????ํ
5__inference_SqueezeNet_Preloaded_layer_call_fn_267270ณU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กข@ข=
6ข3
)&
Input?????????เเ
p 

 
ช "?????????๎
5__inference_SqueezeNet_Preloaded_layer_call_fn_268369ดU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กขAข>
7ข4
*'
inputs?????????เเ
p

 
ช "?????????๎
5__inference_SqueezeNet_Preloaded_layer_call_fn_268478ดU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กขAข>
7ข4
*'
inputs?????????เเ
p 

 
ช "?????????๐
!__inference__wrapped_model_264942สU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กข8ข5
.ข+
)&
Input?????????เเ
ช "7ช4
2

DenseFinal$!

DenseFinal?????????ฌ
E__inference_flatten_1_layer_call_and_return_conditional_losses_269415c8ข5
.ข+
)&
inputs?????????
ช "'ข$

0?????????ค
 
*__inference_flatten_1_layer_call_fn_269420V8ข5
.ข+
)&
inputs?????????
ช "?????????ค;
__inference_loss_fn_0_2694511ข

ข 
ช " =
__inference_loss_fn_10_269561ข

ข 
ช " =
__inference_loss_fn_11_269572ข

ข 
ช " =
__inference_loss_fn_12_269583ข

ข 
ช " =
__inference_loss_fn_13_269594ชข

ข 
ช " =
__inference_loss_fn_14_269605ฑข

ข 
ช " =
__inference_loss_fn_15_269616ธข

ข 
ช " =
__inference_loss_fn_16_269627ฤข

ข 
ช " =
__inference_loss_fn_17_269638หข

ข 
ช " =
__inference_loss_fn_18_269649าข

ข 
ช " =
__inference_loss_fn_19_269660?ข

ข 
ช " ;
__inference_loss_fn_1_269462=ข

ข 
ช " =
__inference_loss_fn_20_269671ๅข

ข 
ช " =
__inference_loss_fn_21_269682์ข

ข 
ช " =
__inference_loss_fn_22_269693?ข

ข 
ช " =
__inference_loss_fn_23_269704ข

ข 
ช " =
__inference_loss_fn_24_269715ข

ข 
ช " ;
__inference_loss_fn_2_269473Dข

ข 
ช " ;
__inference_loss_fn_3_269484Kข

ข 
ช " ;
__inference_loss_fn_4_269495Wข

ข 
ช " ;
__inference_loss_fn_5_269506^ข

ข 
ช " ;
__inference_loss_fn_6_269517eข

ข 
ช " ;
__inference_loss_fn_7_269528qข

ข 
ช " ;
__inference_loss_fn_8_269539xข

ข 
ช " ;
__inference_loss_fn_9_269550ข

ข 
ช " ?
$__inference_signature_wrapper_267537ำU12=>DEKLWX^_efqrxyชซฑฒธนฤลหฬาำ?฿ๅๆ์ํ??กขAข>
ข 
7ช4
2
Input)&
Input?????????เเ"7ช4
2

DenseFinal$!

DenseFinal?????????