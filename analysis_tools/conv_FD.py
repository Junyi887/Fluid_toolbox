'''
Apated from https://github.com/paulpuren/PhySR/blob/main/Code

'''


import torch
import torch.nn as nn   

# https://edisciplinas.usp.br/pluginfile.php/41896/mod_resource/content/1/LeVeque%20Finite%20Diff.pdf

class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) // 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol    

class ConvFD(nn.Module):
    def __init__(self,dx = 0.012,kernel_size = 3):
        super(ConvFD,self).__init__()
        self.delta_x = dx
        # https://en.wikipedia.org/wiki/Finite_difference_coefficient
        self.filter_y4 = [[[[    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0],
           [1/12, -8/12,  0,  8/12, -1/12],
           [    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0]]]]


        self.filter_x4 = [[[[    0,   0,   1/12,   0,     0],
           [    0,   0,   -8/12,   0,     0],
           [    0,   0,   0,   0,     0],
           [    0,   0,   8/12,   0,     0],
           [    0,   0,   -1/12,   0,     0]]]]
        

        self.laplace_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]
        

        self.filter_x2 = [[[[    0,   -1/2,   0],
                    [    0,   0,   0],
                    [     0,   1/2,   0]]]]

        self.filter_y2 = [[[[    0,   0,   0],
                    [    -1/2,   0,   1/2],
                    [     0,   0,   0]]]]
        

        if kernel_size ==5:
            self.dx = Conv2dDerivative(
                DerFilter = self.filter_x4,
                resol = self.delta_x,
                kernel_size = 5,
                name = 'dx_operator').cuda()

            self.dy = Conv2dDerivative(
                DerFilter = self.filter_y4,
                resol = self.delta_x,
                kernel_size = 5,
                name = 'dy_operator').cuda()  
            
            self.laplace = Conv2dDerivative(
                DerFilter = self.laplace_op,
                resol = self.delta_x,
                kernel_size = 5,
                name = 'laplace_operator').cuda()
            
        elif kernel_size ==3:
            self.dx = Conv2dDerivative(
                DerFilter = self.filter_x2,
                resol = self.delta_x,
                kernel_size = 3,
                name = 'dx_operator').cuda()

            self.dy = Conv2dDerivative(
                DerFilter = self.filter_y2,
                resol = self.delta_x,
                kernel_size = 3,
                name = 'dy_operator').cuda()  
            
