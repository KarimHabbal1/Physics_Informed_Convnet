import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint_torch
import numpy as np
from SINDy_functions import jacobian

class SINDy(nn.Module):
    def __init__(self, dt, coefs, input_dim = 2, latent_dim = 2, library_dim = 6, poly_order = 2, l = {'l1': 1e-1, 'l2': 1e-1, 'l3': 1e-1, 'l4': 1e-1, 'l5': 1e-1, 'l6': 1e-1}):

        super(SINDy, self).__init__()
        self.dt = dt
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.library_dim = library_dim
        self.poly_order = poly_order
        self.l = l
        self.coefs = coefs
        self.coefficients = nn.Parameter(coefs)

    def phi(self, x):
        library = [torch.ones(x.size(0), 1)]  # Bias term
        for i in range(self.latent_dim):
            library.append(x[:, i:i+1])  # Linear terms

        if self.poly_order >= 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    library.append((x[:, i] * x[:, j]).unsqueeze(1))  # Second order terms

        # Adding sine terms for each variable
        for i in range(self.latent_dim):
            library.append(torch.sin(x[:, i:i+1]))

        return torch.cat(library, dim=1)

    def phi_t(self, x, t):
        library = [torch.ones(x.size(0), 1)]  # Bias term
        for i in range(self.latent_dim):
            library.append(x[:, i:i+1])  # Linear terms

        if self.poly_order >= 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    library.append((x[:, i] * x[:, j]).unsqueeze(1))  # Second order terms

        # Adding sine terms for each variable
        for i in range(self.latent_dim):
            library.append(torch.sin(x[:, i:i+1]))

        # Incorporating time into the library
        library_with_t = [entry * t.unsqueeze(1) for entry in library]
        library_with_t2 = [entry * torch.pow(t, 2).unsqueeze(1) for entry in library]
        library.extend(library_with_t)
        library.extend(library_with_t2)

        return torch.cat(library, dim=1)

    def SINDy_num(self, x):
        dxdt = torch.matmul(self.phi(x), self.coefficients)
        return dxdt
    
    def SINDy_num_t(self, t, x):
        dxdt = torch.matmul(self.phi_t(x,t), self.coefficients)
        return dxdt
    
    def integrate(self, x0, t):
        try:
            x_pred = odeint_torch(self.SINDy_num, x0, t) 
        except AssertionError as error:
            print(error)
            return None 
        return x_pred
    
    def Loss(self, v, dvdt, criterion):

        with torch.autograd.enable_grad():
            x = self.encoder(v) #ur latent variables 
            v_bar = self.decoder(x)

        time = torch.tensor(np.linspace(0, self.dt*len(x), len(x), endpoint=False),requires_grad=False, dtype=torch.float32)
        time_int = torch.tensor(np.linspace(0, self.dt*self.t_int, self.t_int, endpoint=False),requires_grad=False, dtype=torch.float32) # to be used in integration
            
        loss = 0

        if self.l['l1'] > 0:
            loss += criterion(x[:,0], v[:,0])*self.l['l1']

        if self.l['l2'] > 0:
            loss += criterion(v, v_bar)*self.l['l2']

        if self.l['l3'] > 0 or self.l['l4'] > 0:

            dxdt_SINDy = self.SINDy_num(time, x)

            with torch.autograd.enable_grad():
                dxdv = jacobian(x, v)
            dxdt = torch.einsum('ijk,ij->ik', dxdv[1:], dvdt[1:])

            with torch.autograd.enable_grad():
                dvdx = jacobian(v_bar, x)
            dvdt_dec = torch.einsum('ijk,ij->ik', dvdx[1:], dxdt_SINDy[1:])

            loss += criterion(dxdt, dxdt_SINDy[1:])*self.l['l3']
            loss += criterion(dvdt_dec, dvdt[1:])*self.l['l4']

        if self.l['l5'] > 0:
            loss += torch.norm(self.coefficients, 1)*self.l['l5']

        if self.l['l6'] > 0:
            sol = x[:len(time_int)] 
            loss += criterion(sol[:, 0], v[:, 0][:len(sol[:,0])])
            for j in range(self.latent_dim - 1):
                for i in range(1, self.tau + 1):
                    k1 = self.SINDy_num(time_int, sol)
                    k2 = self.SINDy_num(time_int, sol + self.dt/2 * k1)
                    k3 = self.SINDy_num(time_int, sol + self.dt/2 * k2)
                    k4 = self.SINDy_num(time_int, sol + self.dt * k3)
                    sol = sol + 1/6 * self.dt * (k1 + 2*k2 + 2*k3 + k4) 
                loss += criterion(sol[:,0], v[:,j+1][:len(sol[:,0])])

        
        return loss
    