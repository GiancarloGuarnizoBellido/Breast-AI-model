
import torch
import torch.nn as nn
import torch.nn.functional as F



class conv_block_encoder(nn.Module): #subclass of nn.Module, give the subclass some functionalities and properties of PyTorch.
    def __init__(self,input_channels,n_features):
        #input_channels can be 1 (grey scale) or 3 (rgb color); the conv with a kernel of n features, results on an image with channels equals to n_features
        super().__init__() #call the constructor (__init__) of nn.Module for the correct implementation of PyTorch

        #nn.Sequential to pack and define a sequence of layers
        self.conv_result=nn.Sequential(
            nn.Conv3d(input_channels,n_features,kernel_size=(3,3,3),padding="same",bias=False),#the resulting image will have the same size as the input one
            nn.InstanceNorm3d(n_features),
            #nn.BatchNorm3d(n_features), #For better performance, has a proper bias, so previously ther's no need to have one (bias=False)
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.SiLU(inplace=True), #Sigmoid Linear Unit
            #nn.ReLU(inplace=True),#activation function, it prevents linearity so the u-net learn complex patrons
            #if the value of a pixel is negative, it becomes 0, otherwise, lets pass the positives values
            #In the U-net, its applied two times
            #nn.Dropout(0.3),#Randomly drop nodes to prevent overfitting
            nn.Conv3d(n_features,n_features,kernel_size=(3,3,3),padding="same",bias=False),
            nn.InstanceNorm3d(n_features),
            #nn.BatchNorm3d(n_features),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.SiLU(inplace=True), #Sigmoid Linear Unit
            nn.Dropout(0.3),
        )
    def forward(self,x): #Instruct how the sequence is applied, is called implicitly in the model
        return self.conv_result(x)
    
class conv_block_decoder(nn.Module): #subclass of nn.Module, give the subclass some functionalities and properties of PyTorch.
    def __init__(self,input_channels,n_features):
        #input_channels can be 1 (grey scale) or 3 (rgb color); the conv with a kernel of n features, results on an image with channels equals to n_features
        super().__init__() #call the constructor (__init__) of nn.Module for the correct implementation of PyTorch

        #nn.Sequential to pack and define a sequence of layers
        self.conv_result=nn.Sequential(
            nn.Conv3d(input_channels,n_features,kernel_size=(3,3,3),padding="same",bias=False),#the resulting image will have the same size as the input one
            nn.InstanceNorm3d(n_features),
            #nn.BatchNorm3d(n_features), #For better performance, has a proper bias, so previously ther's no need to have one (bias=False)
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.SiLU(inplace=True), #Sigmoid Linear Unit
            #nn.ReLU(inplace=True),#activation function, it prevents linearity so the u-net learn complex patrons
            #if the value of a pixel is negative, it becomes 0, otherwise, lets pass the positives values
            #In the U-net, its applied two times
            #nn.Dropout(0.3),#Randomly drop nodes to prevent overfitting
            nn.Conv3d(n_features,n_features,kernel_size=(3,3,3),padding="same",bias=False),
            nn.InstanceNorm3d(n_features),
            #nn.BatchNorm3d(n_features),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.SiLU(inplace=True), #Sigmoid Linear Unit
            nn.Dropout(0.3),
        )
    def forward(self,x): #Instruct how the sequence is applied, is called implicitly in the model
        return self.conv_result(x)

class encoder_block(nn.Module): #Consist in two conv of 3x3 kernel and a 2x2 max pooling with stride value of 2
    def __init__(self,input_channels,n_features):
        super().__init__()
        self.conv_result=conv_block_encoder(input_channels,n_features) #two conv of 3x3 kernel
        self.max_pool_result=nn.MaxPool3d(kernel_size=(2,2,2), stride=2,ceil_mode=True) #2x2 max pooling with stride value of 2

    def forward(self,x):
        fx=self.conv_result(x) #signal will be use in the attention gate
        ouput=self.max_pool_result(fx) #ouput will be use for the next procedures of the u-net
        return fx,ouput

class decoder_block(nn.Module): #Consist in an upsampling, concatenation of spacial and feature information, and convolution; before concatenation an attention gate is applied
    def __init__(self,input_channels,n_features):
        super().__init__()
        self.up_result=nn.Upsample(scale_factor=2,mode="trilinear",align_corners=True) #The size of the image input is duplicated by twos
        #self.up_result=nn.ConvTranspose3d(input_channels[0], input_channels[0], kernel_size=(2,2,2), stride=2)
        #Puede aprender los pesos del kernel durante el entrenamiento, ajustando los parámetros de upsampling
        self.ag_result=attention_gate(input_channels,n_features) #Give the spacial information with the corresponding weights
        self.conv_result=conv_block_decoder(input_channels[0]+input_channels[1],n_features) #Conv of 3x3 kernel

    def forward(self,x,s):
        y=self.ag_result(x,s) #The spacial information is provided with weights
        x_up=self.up_result(x) #First upsampling
        #print(f"mult: {y.shape}")
        #print(f"x: {x.shape}")
        #print(f"concat: {torch.cat((x,y),dim=1).shape}")
        x_out=self.conv_result(torch.cat((x_up,y),dim=1)) #Apply the two conv to the concatenation of the feature and spacial information
        #print(f"x_out: {x_out.shape}")
        return x_out


class attention_gate(nn.Module): #In the decoding phase, we use the spacial information from the encoding phase with weights defined by the u-net
    def __init__(self,input_channels,n_features): #input_channels has two values: s,p
        super().__init__()
        #Obtain the weight with the same size of features
        #Weights obtention of spacial information
        self.weight_g=nn.Sequential(
            nn.Conv3d(input_channels[0],n_features,kernel_size=(1,1,1), stride=1, bias=False),
            nn.InstanceNorm3d(n_features)
            #nn.BatchNorm3d(n_features)
        )
        #Weights obtention of feature information
        self.weight_fx=nn.Sequential(
            nn.Conv3d(input_channels[1],n_features,kernel_size=(1,1,1), stride=2, bias=False),
            nn.InstanceNorm3d(n_features)
            #nn.BatchNorm3d(n_features)
        )
        #ReLU to prevent lineality
        #self.relu=nn.ReLU(inplace=True)      
        self.relu=nn.LeakyReLU(negative_slope=0.01, inplace=True)
        #self.relu=nn.SiLU(inplace=True) #Sigmoid Linear Unit
        #Conv to obtain 1 feature and then a sigmoid function
        self.output=nn.Sequential(
            nn.Conv3d(n_features,1,kernel_size=(1,1,1), padding="same", stride=1, bias=False),
            nn.InstanceNorm3d(1),
            #nn.BatchNorm3d(1),
            nn.Sigmoid() #Sigmoid is used to scale all the values from 0 to 1
        )

    def forward(self,g,fx):
        #First obtain the weights with the same dimensions
        #print(g.shape)
        #print(fx.shape)
        weight_g=self.weight_g(g)
        weight_fx=self.weight_fx(fx)
        #The sum of the weights pass through a relu activation. The sum is element by element, the dimension is not alterated
        out=self.relu(weight_g+weight_fx) #The weights can go anywhere from 0 to infinite
        out=self.output(out)
        #out = F.upsample(out, size=fx.size()[2:], mode='trilinear')
        out = F.interpolate(out, size=fx.size()[2:], mode='trilinear')

        #print(out.shape)
        return out*fx #Upsampling to the original size of x, from 1 feature pass to n_feature
    
class attention_unet_seg(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.e1=encoder_block(input_channels,16) #the 64 value is the number of feature which is defined in the u-net arquitecture
        #The resulting image will have a number of channels (input_channels) equals to the number of features defined in the kernel
        self.e2=encoder_block(16,32) #64 -> 128
        self.e3=encoder_block(32,64) #128 -> 256
        self.e4=encoder_block(64,128) ##256 -> 512
        
        #The bridge
        self.b1=conv_block_encoder(128,256) # 512 -> 1024
        #Now the procedure is the inverse, so each pass in the decoder divide by two the number of features
        #[a,b]: a: features of the gating signal; b: features of the input from encoder
        self.d1=decoder_block([256,128],128)  #1024 -> 512
        self.d2=decoder_block([128,64],64)  #512 -> 256
        self.d3=decoder_block([64,32],32)  #256 -> 128
        self.d4=decoder_block([32,16],16) #128 -> 64

        self.output=nn.Conv3d(16,1,kernel_size=(1,1,1),padding="same")

    def forward(self,x):
        #Encoder
        fx_1,p1=self.e1(x) #self.e contains two values in encoder_block class
        #print(f"a: {fx_1.shape}")
        fx_2,p2=self.e2(p1)
        #print(f"b: {fx_2.shape}")
        fx_3,p3=self.e3(p2)
        #print(f"c: {fx_3.shape}")
        fx_4,p4=self.e4(p3)
        #print(f"d: {fx_4.shape}")
        #Bridge
        b1=self.b1(p4)
        #print(f"e: {b1.shape}")
        #Decoder
        d1=self.d1(b1,fx_4)
        d2=self.d2(d1,fx_3)
        d3=self.d3(d2,fx_2)
        d4=self.d4(d3,fx_1)
        #Final operation
        out=self.output(d4)
        return out
    














# class attention_unet_512_feat(nn.Module):
#     def __init__(self,input_channels):
#         super().__init__()
#         self.e1=encoder_block(input_channels,16) #the 64 value is the number of feature which is defined in the u-net arquitecture
#         #The resulting image will have a number of channels (input_channels) equals to the number of features defined in the kernel
#         self.e2=encoder_block(16,32) #15 -> 32
#         self.e3=encoder_block(32,64) #32 -> 64
#         self.e4=encoder_block(64,128) ##64 -> 128
#         self.e5=encoder_block(128,256) ##128 -> 256
#         #The bridge
#         self.b1=conv_block_encoder(256,512) # 512 -> 1024
#         #Now the procedure is the inverse, so each pass in the decoder divide by two the number of features
#         #[a,b]: a: features of the gating signal; b: features of the input from encoder
        
#         self.d1=decoder_block([512,256],256) #128 -> 64
#         self.d2=decoder_block([256,128],128)  #1024 -> 512
#         self.d3=decoder_block([128,64],64)  #512 -> 256
#         self.d4=decoder_block([64,32],32)  #256 -> 128
#         self.d5=decoder_block([32,16],16) #128 -> 64
#         self.output=nn.Conv3d(16,1,kernel_size=(1,1,1),padding="same")

#     def forward(self,x):
#         #Encoder
#         fx_1,p1=self.e1(x) #self.e contains two values in encoder_block class
#         #print(f"a: {fx_1.shape}")
#         fx_2,p2=self.e2(p1)
#         #print(f"b: {fx_2.shape}")
#         fx_3,p3=self.e3(p2)
#         #print(f"c: {fx_3.shape}")
#         fx_4,p4=self.e4(p3)
#         #print(f"d: {fx_4.shape}")
#         fx_5,p5=self.e5(p4)
#         #Bridge
#         b1=self.b1(p5)
#         #print(f"e: {b1.shape}")
#         #Decoder
#         d1=self.d1(b1,fx_5)
#         d2=self.d2(d1,fx_4)
#         d3=self.d3(d2,fx_3)
#         d4=self.d4(d3,fx_2)
#         d5=self.d5(d4,fx_1)
#         #Final operation
#         out=self.output(d5)
#         return out
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class conv_block(nn.Module): #subclass of nn.Module, give the subclass some functionalities and properties of PyTorch.
#     def __init__(self,input_channels,n_features):
#         #input_channels can be 1 (grey scale) or 3 (rgb color); the conv with a kernel of n features, results on an image with channels equals to n_features
#         super().__init__() #call the constructor (__init__) of nn.Module for the correct implementation of PyTorch

#         #nn.Sequential to pack and define a sequence of layers
#         self.conv_result=nn.Sequential(
#             nn.Conv3d(input_channels,n_features,kernel_size=(3,3,3),padding="same",bias=False),#the resulting image will have the same size as the input one
#             nn.InstanceNorm3d(n_features),
#             #nn.BatchNorm3d(n_features), #For better performance, has a proper bias, so previously ther's no need to have one (bias=False)
#             #nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.SiLU(inplace=True), #Sigmoid Linear Unit
#             #nn.ReLU(inplace=True),#activation function, it prevents linearity so the u-net learn complex patrons
#             #if the value of a pixel is negative, it becomes 0, otherwise, lets pass the positives values
#             #In the U-net, its applied two times
#             #nn.Dropout(0.2),#Randomly drop nodes to prevent overfitting
#             nn.Conv3d(n_features,n_features,kernel_size=(3,3,3),padding="same",bias=False),
#             nn.InstanceNorm3d(n_features),
#             #nn.BatchNorm3d(n_features),
#             #nn.ReLU(inplace=True),
#             #nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.SiLU(inplace=True), #Sigmoid Linear Unit
#             nn.Dropout(0.3),
#         )
#     def forward(self,x): #Instruct how the sequence is applied, is called implicitly in the model
#         return self.conv_result(x)

# class encoder_block(nn.Module): #Consist in two conv of 3x3 kernel and a 2x2 max pooling with stride value of 2
#     def __init__(self,input_channels,n_features):
#         super().__init__()
#         self.conv_result=conv_block(input_channels,n_features) #two conv of 3x3 kernel
#         self.max_pool_result=nn.MaxPool3d(kernel_size=(2,2,2), stride=2,ceil_mode=True) #2x2 max pooling with stride value of 2

#     def forward(self,x):
#         fx=self.conv_result(x) #signal will be use in the attention gate
#         ouput=self.max_pool_result(fx) #ouput will be use for the next procedures of the u-net
#         return fx,ouput

# class decoder_block(nn.Module): #Consist in an upsampling, concatenation of spacial and feature information, and convolution; before concatenation an attention gate is applied
#     def __init__(self,input_channels,n_features):
#         super().__init__()
#         self.up_result=nn.Upsample(scale_factor=2,mode="trilinear",align_corners=True) #The size of the image input is duplicated by twos
#         #self.up_result=nn.ConvTranspose3d(input_channels[0], input_channels[0], kernel_size=(2,2,2), stride=2)
#         #Puede aprender los pesos del kernel durante el entrenamiento, ajustando los parámetros de upsampling
#         self.ag_result=attention_gate(input_channels,n_features) #Give the spacial information with the corresponding weights
#         self.conv_result=conv_block(input_channels[0]+input_channels[1],n_features) #Conv of 3x3 kernel

#     def forward(self,x,s):
#         y=self.ag_result(x,s) #The spacial information is provided with weights
#         x_up=self.up_result(x) #First upsampling
#         #print(f"mult: {y.shape}")
#         #print(f"x: {x.shape}")
#         #print(f"concat: {torch.cat((x,y),dim=1).shape}")
#         x_out=self.conv_result(torch.cat((x_up,y),dim=1)) #Apply the two conv to the concatenation of the feature and spacial information
#         #print(f"x_out: {x_out.shape}")
#         return x_out


# class attention_gate(nn.Module): #In the decoding phase, we use the spacial information from the encoding phase with weights defined by the u-net
#     def __init__(self,input_channels,n_features): #input_channels has two values: s,p
#         super().__init__()
#         #Obtain the weight with the same size of features
#         #Weights obtention of spacial information
#         self.weight_g=nn.Sequential(
#             nn.Conv3d(input_channels[0],n_features,kernel_size=(1,1,1), stride=1, bias=False),
#             nn.InstanceNorm3d(n_features)
#             #nn.BatchNorm3d(n_features)
#         )
#         #Weights obtention of feature information
#         self.weight_fx=nn.Sequential(
#             nn.Conv3d(input_channels[1],n_features,kernel_size=(1,1,1), stride=2, bias=False),
#             nn.InstanceNorm3d(n_features)
#             #nn.BatchNorm3d(n_features)
#         )
#         #ReLU to prevent lineality
#         #self.relu=nn.LeakyReLU(negative_slope=0.01, inplace=True)
#         #self.relu=nn.ReLU(inplace=True) #Sigmoid Linear Unit
#         self.relu=nn.SiLU(inplace=True) #Sigmoid Linear Unit
#         #Conv to obtain 1 feature and then a sigmoid function
#         self.output=nn.Sequential(
#             nn.Conv3d(n_features,1,kernel_size=(1,1,1), padding="same", stride=1, bias=False),
#             nn.InstanceNorm3d(1),
#             #nn.BatchNorm3d(1),
#             nn.Sigmoid() #Sigmoid is used to scale all the values from 0 to 1
#         )

#     def forward(self,g,fx):
#         #First obtain the weights with the same dimensions
#         #print(g.shape)
#         #print(fx.shape)
#         weight_g=self.weight_g(g)
#         weight_fx=self.weight_fx(fx)
#         #The sum of the weights pass through a relu activation. The sum is element by element, the dimension is not alterated
#         out=self.relu(weight_g+weight_fx) #The weights can go anywhere from 0 to infinite
#         out=self.output(out)
#         #out = F.upsample(out, size=fx.size()[2:], mode='trilinear')
#         out = F.interpolate(out, size=fx.size()[2:], mode='trilinear')

#         #print(out.shape)
#         return out*fx #Upsampling to the original size of x, from 1 feature pass to n_feature

# class attention_unet_seg(nn.Module):
#     def __init__(self,input_channels):
#         super().__init__()
#         self.e1=encoder_block(input_channels,16) #the 64 value is the number of feature which is defined in the u-net arquitecture
#         #The resulting image will have a number of channels (input_channels) equals to the number of features defined in the kernel
#         self.e2=encoder_block(16,32) #64 -> 128
#         self.e3=encoder_block(32,64) #128 -> 256
#         self.e4=encoder_block(64,128) ##256 -> 512
        
#         #The bridge
#         self.b1=conv_block(128,256) # 512 -> 1024
#         #Now the procedure is the inverse, so each pass in the decoder divide by two the number of features
#         #[a,b]: a: features of the gating signal; b: features of the input from encoder
#         self.d1=decoder_block([256,128],128)  #1024 -> 512
#         self.d2=decoder_block([128,64],64)  #512 -> 256
#         self.d3=decoder_block([64,32],32)  #256 -> 128
#         self.d4=decoder_block([32,16],16) #128 -> 64

#         self.output=nn.Conv3d(16,1,kernel_size=(1,1,1),padding="same")

#     def forward(self,x):
#         #Encoder
#         fx_1,p1=self.e1(x) #self.e contains two values in encoder_block class
#         #print(f"a: {fx_1.shape}")
#         fx_2,p2=self.e2(p1)
#         #print(f"b: {fx_2.shape}")
#         fx_3,p3=self.e3(p2)
#         #print(f"c: {fx_3.shape}")
#         fx_4,p4=self.e4(p3)
#         #print(f"d: {fx_4.shape}")
#         #Bridge
#         b1=self.b1(p4)
#         #print(f"e: {b1.shape}")
#         #Decoder
#         d1=self.d1(b1,fx_4)
#         d2=self.d2(d1,fx_3)
#         d3=self.d3(d2,fx_2)
#         d4=self.d4(d3,fx_1)
#         #Final operation
#         out=self.output(d4)
#         return out

# class attention_unet_seg_v1(nn.Module):
#     def __init__(self,input_channels):
#         super().__init__()
#         self.e1=encoder_block(input_channels,16) #the 64 value is the number of feature which is defined in the u-net arquitecture
#         #The resulting image will have a number of channels (input_channels) equals to the number of features defined in the kernel
#         self.e2=encoder_block(16,32) #64 -> 128
#         self.e3=encoder_block(32,64) #128 -> 256
#         self.e4=encoder_block(64,128) ##256 -> 512
#         self.e5=encoder_block(128,256) ##256 -> 512
       
#         #The bridge
#         self.b1=conv_block(256,512) # 512 -> 1024

#         #Now the procedure is the inverse, so each pass in the decoder divide by two the number of features
#         #[a,b]: a: features of the gating signal; b: features of the input from encoder
#         self.d1=decoder_block([512,256],256)
#         self.d2=decoder_block([256,128],128)  #1024 -> 512
#         self.d3=decoder_block([128,64],64)  #512 -> 256
#         self.d4=decoder_block([64,32],32)  #256 -> 128
#         self.d5=decoder_block([32,16],16)  #128 -> 64

#         self.output=nn.Conv3d(16,1,kernel_size=(1,1,1),padding=0)

#     def forward(self,x):
#         #Encoder
#         fx_1,p1=self.e1(x) #self.e contains two values in encoder_block class
#         #print(f"a: {fx_1.shape}")
#         fx_2,p2=self.e2(p1)
#         #print(f"b: {fx_2.shape}")
#         fx_3,p3=self.e3(p2)
#         #print(f"c: {fx_3.shape}")
#         fx_4,p4=self.e4(p3)
#         #print(f"d: {fx_4.shape}")
#         fx_5,p5=self.e5(p4)
       
#         #Bridge
#         b1=self.b1(p5)
#         #print(f"e: {b1.shape}")
#         #Decoder
#         d1=self.d1(b1,fx_5)
#         d2=self.d2(d1,fx_4)
#         d3=self.d3(d2,fx_3)
#         d4=self.d4(d3,fx_2)
#         d5=self.d5(d4,fx_1)
        
#         d6 = self.output(d5)  # Promedio final
#         #out = torch.sigmoid(d6)  # Obtener probabilidad
#         return d6









# import torch
# import torch.nn as nn
# import torch.nn.functional as F



# class conv_block(nn.Module): #subclass of nn.Module, give the subclass some functionalities and properties of PyTorch.
#     def __init__(self,input_channels,n_features):
#         #input_channels can be 1 (grey scale) or 3 (rgb color); the conv with a kernel of n features, results on an image with channels equals to n_features
#         super().__init__() #call the constructor (__init__) of nn.Module for the correct implementation of PyTorch

#         #nn.Sequential to pack and define a sequence of layers
#         self.conv_result=nn.Sequential(
#             nn.Conv3d(input_channels,n_features,kernel_size=(3,3,3),padding="same",bias=False),#the resulting image will have the same size as the input one
#             nn.InstanceNorm3d(n_features),
#             #nn.BatchNorm3d(n_features), #For better performance, has a proper bias, so previously ther's no need to have one (bias=False)
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             #nn.SiLU(inplace=True), #Sigmoid Linear Unit
#             #nn.ReLU(inplace=True),#activation function, it prevents linearity so the u-net learn complex patrons
#             #if the value of a pixel is negative, it becomes 0, otherwise, lets pass the positives values
#             #In the U-net, its applied two times
#             nn.Dropout(0.3),#Randomly drop nodes to prevent overfitting
#             nn.Conv3d(n_features,n_features,kernel_size=(3,3,3),padding="same",bias=False),
#             nn.InstanceNorm3d(n_features),
#             #nn.BatchNorm3d(n_features),
#             #nn.ReLU(inplace=True),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             #nn.SiLU(inplace=True), #Sigmoid Linear Unit
#             nn.Dropout(0.2),
#         )
#     def forward(self,x): #Instruct how the sequence is applied, is called implicitly in the model
#         return self.conv_result(x)

# class encoder_block(nn.Module): #Consist in two conv of 3x3 kernel and a 2x2 max pooling with stride value of 2
#     def __init__(self,input_channels,n_features):
#         super().__init__()
#         self.conv_result=conv_block(input_channels,n_features) #two conv of 3x3 kernel
#         self.max_pool_result=nn.MaxPool3d(kernel_size=(2,2,2), stride=2,ceil_mode=True) #2x2 max pooling with stride value of 2

#     def forward(self,x):
#         fx=self.conv_result(x) #signal will be use in the attention gate
#         ouput=self.max_pool_result(fx) #ouput will be use for the next procedures of the u-net
#         return fx,ouput

# class decoder_block(nn.Module): #Consist in an upsampling, concatenation of spacial and feature information, and convolution; before concatenation an attention gate is applied
#     def __init__(self,input_channels,n_features):
#         super().__init__()
#         self.up_result=nn.Upsample(scale_factor=2,mode="trilinear",align_corners=True) #The size of the image input is duplicated by twos
#         #self.up_result=nn.ConvTranspose3d(input_channels[0], input_channels[0], kernel_size=(2,2,2), stride=2)
#         #Puede aprender los pesos del kernel durante el entrenamiento, ajustando los parámetros de upsampling
#         self.ag_result=attention_gate(input_channels,n_features) #Give the spacial information with the corresponding weights
#         self.conv_result=conv_block(input_channels[0]+input_channels[1],n_features) #Conv of 3x3 kernel

#     def forward(self,x,s):
#         y=self.ag_result(x,s) #The spacial information is provided with weights
#         x_up=self.up_result(x) #First upsampling
#         #print(f"mult: {y.shape}")
#         #print(f"x: {x.shape}")
#         #print(f"concat: {torch.cat((x,y),dim=1).shape}")
#         x_out=self.conv_result(torch.cat((x_up,y),dim=1)) #Apply the two conv to the concatenation of the feature and spacial information
#         #print(f"x_out: {x_out.shape}")
#         return x_out


# class attention_gate(nn.Module): #In the decoding phase, we use the spacial information from the encoding phase with weights defined by the u-net
#     def __init__(self,input_channels,n_features): #input_channels has two values: s,p
#         super().__init__()
#         #Obtain the weight with the same size of features
#         #Weights obtention of spacial information
#         self.weight_g=nn.Sequential(
#             nn.Conv3d(input_channels[0],n_features,kernel_size=(1,1,1), stride=1, bias=False),
#             nn.InstanceNorm3d(n_features)
#         )
#         #Weights obtention of feature information
#         self.weight_fx=nn.Sequential(
#             nn.Conv3d(input_channels[1],n_features,kernel_size=(1,1,1), stride=2, bias=False),
#             nn.InstanceNorm3d(n_features)
#         )
#         #ReLU to prevent lineality
#         self.relu=nn.LeakyReLU(negative_slope=0.01, inplace=True)
#         #self.relu=nn.SiLU(inplace=True) #Sigmoid Linear Unit
#         #Conv to obtain 1 feature and then a sigmoid function
#         self.output=nn.Sequential(
#             nn.Conv3d(n_features,1,kernel_size=(1,1,1), padding="same", stride=1, bias=False),
#             nn.InstanceNorm3d(1),
#             nn.Sigmoid() #Sigmoid is used to scale all the values from 0 to 1
#         )

#     def forward(self,g,fx):
#         #First obtain the weights with the same dimensions
#         #print(g.shape)
#         #print(fx.shape)
#         weight_g=self.weight_g(g)
#         weight_fx=self.weight_fx(fx)
#         #The sum of the weights pass through a relu activation. The sum is element by element, the dimension is not alterated
#         out=self.relu(weight_g+weight_fx) #The weights can go anywhere from 0 to infinite
#         out=self.output(out)
#         #out = F.upsample(out, size=fx.size()[2:], mode='trilinear')
#         out = F.interpolate(out, size=fx.size()[2:], mode='trilinear')

#         #print(out.shape)
#         return out*fx #Upsampling to the original size of x, from 1 feature pass to n_feature

# class attention_unet(nn.Module):
#     def __init__(self,input_channels):
#         super().__init__()
#         self.e1=encoder_block(input_channels,16) #the 64 value is the number of feature which is defined in the u-net arquitecture
#         #The resulting image will have a number of channels (input_channels) equals to the number of features defined in the kernel
#         self.e2=encoder_block(16,32) #15 -> 32
#         self.e3=encoder_block(32,64) #32 -> 64
#         self.e4=encoder_block(64,128) ##64 -> 128
#         self.e5=encoder_block(128,256) ##128 -> 256
#         #The bridge
#         self.b1=conv_block(256,512) # 512 -> 1024
#         #Now the procedure is the inverse, so each pass in the decoder divide by two the number of features
#         #[a,b]: a: features of the gating signal; b: features of the input from encoder
        
#         self.d1=decoder_block([512,256],256) #128 -> 64
#         self.d2=decoder_block([256,128],128)  #1024 -> 512
#         self.d3=decoder_block([128,64],64)  #512 -> 256
#         self.d4=decoder_block([64,32],32)  #256 -> 128
#         self.d5=decoder_block([32,16],16) #128 -> 64
#         self.output=nn.Conv3d(16,1,kernel_size=(1,1,1),padding="same")

#     def forward(self,x):
#         #Encoder
#         fx_1,p1=self.e1(x) #self.e contains two values in encoder_block class
#         #print(f"a: {fx_1.shape}")
#         fx_2,p2=self.e2(p1)
#         #print(f"b: {fx_2.shape}")
#         fx_3,p3=self.e3(p2)
#         #print(f"c: {fx_3.shape}")
#         fx_4,p4=self.e4(p3)
#         #print(f"d: {fx_4.shape}")
#         fx_5,p5=self.e5(p4)
#         #Bridge
#         b1=self.b1(p5)
#         #print(f"e: {b1.shape}")
#         #Decoder
#         d1=self.d1(b1,fx_5)
#         d2=self.d2(d1,fx_4)
#         d3=self.d3(d2,fx_3)
#         d4=self.d4(d3,fx_2)
#         d5=self.d5(d4,fx_1)
#         #Final operation
#         out=self.output(d5)
#         return out
    
# class attention_unet_v1(nn.Module):
#     def __init__(self,input_channels):
#         super().__init__()
#         self.e1=encoder_block(input_channels,16) #the 64 value is the number of feature which is defined in the u-net arquitecture
#         #The resulting image will have a number of channels (input_channels) equals to the number of features defined in the kernel
#         self.e2=encoder_block(16,32) #64 -> 128
#         self.e3=encoder_block(32,64) #128 -> 256
#         self.e4=encoder_block(64,128) ##256 -> 512
        
#         #The bridge
#         self.b1=conv_block(128,256) # 512 -> 1024
#         #Now the procedure is the inverse, so each pass in the decoder divide by two the number of features
#         #[a,b]: a: features of the gating signal; b: features of the input from encoder
#         self.d1=decoder_block([256,128],128)  #1024 -> 512
#         self.d2=decoder_block([128,64],64)  #512 -> 256
#         self.d3=decoder_block([64,32],32)  #256 -> 128
#         self.d4=decoder_block([32,16],16) #128 -> 64

#         self.output=nn.Conv3d(16,1,kernel_size=(1,1,1),padding="same")

#     def forward(self,x):
#         #Encoder
#         fx_1,p1=self.e1(x) #self.e contains two values in encoder_block class
#         #print(f"a: {fx_1.shape}")
#         fx_2,p2=self.e2(p1)
#         #print(f"b: {fx_2.shape}")
#         fx_3,p3=self.e3(p2)
#         #print(f"c: {fx_3.shape}")
#         fx_4,p4=self.e4(p3)
#         #print(f"d: {fx_4.shape}")
#         #Bridge
#         b1=self.b1(p4)
#         #print(f"e: {b1.shape}")
#         #Decoder
#         d1=self.d1(b1,fx_4)
#         d2=self.d2(d1,fx_3)
#         d3=self.d3(d2,fx_2)
#         d4=self.d4(d3,fx_1)
#         #Final operation
#         out=self.output(d4)
#         return out