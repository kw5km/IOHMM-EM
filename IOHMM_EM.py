# -*- coding: utf-8 -*-

#!/usr/bin/python
import numpy as np
import scipy.stats as st
import scipy.misc as sm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class IOHMM_EM:
    
    def __init__(self, M, N, D, O=0, num_iters=50, emission_distr='gaussian', hidden_inputs=0, load=None, save=None):
        self.M = M
        self.N = N
        self.O = O
        self.O2 = D.shape[1]-1
        self.num_iters = num_iters
        self.emission_distr = emission_distr
        self.hidden_inputs=hidden_inputs
        self.load=load
        self.save=save
        
        # Extract inputs (u) and outputs (x) from Data
        # D ~ (sequence length, input/hidden/output, number sequences)
        if not hidden_inputs:
            self.us_all = D[:,0,:].astype(np.int8)
            self.xs_all = D[:,1:,:]
        else:
            self.xs_all = D[:,1:,:]
            
        # Total timesteps:
        self.T = D.shape[0]
        
        self.initialize_params()
    

    def initialize_params(self):
        
        N,M,O,O2 = self.N,self.M,self.O, self.O2
        
        if self.load!= 'None':
            self.A= np.load('/u/kw5km/Research/Antibiotic_Resistance/Learned_Params/A_'+self.load+'_'+str(self.N)+'.npy')
            self.B= np.load('/u/kw5km/Research/Antibiotic_Resistance/Learned_Params/B_'+self.load+'_'+str(self.N)+'.npy')
            self.pi= np.load('/u/kw5km/Research/Antibiotic_Resistance/Learned_Params/pi_'+self.load+'_'+str(self.N)+'.npy')
        
        else:
            if self.emission_distr == 'gaussian':
                
                # hold = np.array([np.linspace(np.min(self.xs_all),np.max(self.xs_all),N),0.1*np.ones((N,))])
                # self.B = np.ones((2,N,O2))*hold[:,:,None]
                
                hold1 = np.array([[3*(0+0), 3*(1+0), 3*(0+1), 3*(0+0), 3*(1+1),3*(1+0), 3*(0+1), 3*(1+1)], np.ones((N,))])
                hold2 =  np.array([[2*(0+0), 2*(0+0), 2*(1+0), 2*(0+1), 2*(1+0),2*(0+1), 2*(1+1), 2*(1+1)], np.ones((N,))])
                
                # hold1 = np.array([[0,3.5,3,0,5,3.5,3,5],np.ones((N,))])
                # hold2 = np.array([[0,0,3,3.5,3,3.5,5,5],np.ones((N,))])
                
                self.B = np.ones((2,N,O2))
                self.B[:,:,0] = hold1
                self.B[:,:,1] = hold2

            if self.emission_distr == 'categorical':
                self.B = np.ones((N, O))/O

            self.A = np.ones((N, N, M))/N
            if M>1:
                self.A = self.A+((-.5-.5)*np.random.random(self.A.shape)+.75)
                self.A[self.A<=0]=.01
                for j in range(N):
                    for u in range(M):
                        self.A[j,:,u] = self.A[j,:,u]/np.sum(self.A[j,:,u])
    
            # self.pi = np.ones((N,))/N
            self.pi = .99*np.ones((N,))
            self.pi[1:]=(1-.99)/(N-1) 
        
      
    def calc_B(self, state, x_t):
    
    ### Calcs the emission probability of x_t given y_t
        
        emiss = 0
        
        for i in range(self.O2):
        
            if self.emission_distr == 'gaussian':

                emiss += st.norm.logpdf(x_t[i], self.B[0,state,i], np.sqrt(self.B[1,state,i]))
    
            elif self.emission_distr == 'categorical':
                emiss += np.log(self.B[state, x_t[i]])
        
        return emiss
        
    
    def calc_alph(self, t, state, a_t_prev, x_t):
        
    ### calculates alpha(y_t=state)
    
        if not self.hidden_inputs:
            
            vals = [a_t_prev[i] for i in range(self.N)] + [np.log(self.A[i, state, self.us[t-1]]) for i in range(self.N)]
            a_t = sm.logsumexp(vals)
            
          
        else:
            vals = [a_t_prev[i] for i in range(self.N)] + [np.log(self.A[i, state, l]) for i in range(self.N) for l in range(self.M)]
            a_t = sm.logsumexp(vals)
            
        return a_t + self.calc_B(state, x_t)
    
    
    def calc_beta(self, t, state, b_t_prev, x_t_prev):
        
    ### Calcs beta(y_t=state)
        
        if not self.hidden_inputs:

            vals = [b_t_prev[i] for i in range(self.N)] + [self.calc_B(i , x_t_prev) for i in range(self.N)] + [np.log(self.A[state,i,self.us[t]]) for i in range(self.N)]
            return sm.logsumexp(vals)

        else:
            
            vals = [b_t_prev[i] for i in range(self.N)] + [self.calc_B(i , x_t_prev) for i in range(self.N)] + [np.log(self.A[state,i,self.us[t]]) for i in range(self.N) for l in range(self.M)]
            return sm.logsumexp(vals)
    

    def forward_backward(self):

        alphas = np.zeros((self.T,self.N))
        
        # Loop through emission measurements:
        for t, x_t in enumerate(self.xs):
            
            # Loop through possible hidden states:
            for state in range(self.N):
                
                if t==0:
                    alphas[t,state] = np.log(self.pi[state])+self.calc_B(state,x_t)
                    
                else:
                    alphas[t,state] = self.calc_alph(t,state,alphas[t-1,:],x_t)
            
        betas = np.zeros((self.T,self.N))
        
        # Loop through emission measurements reversed, as beta calc is recursive backwards:
        for t in reversed(range(self.T)):

            for state in range(self.N):
                
                if t == self.T-1: 
                    betas[t,state] = np.log(1)
                    
                else: 
                    betas[t,state] = self.calc_beta(t, state, betas[t+1,:], self.xs[t+1])   
        
        gamma = (alphas+betas)-sm.logsumexp(alphas+betas, axis=1)[:,None]
                        
        return alphas, betas, gamma

    def zeta_calc(self):
        
        alphas, betas, gamma = self.forward_backward()
        
        zeta = np.zeros((self.T-1, self.N, self.N,self.M))

        for t, x_t in enumerate(self.xs[:-1]):
            for j in range(self.N):
                for i in range(self.N):
                    zeta[t, j, i, self.us[t]] = alphas[t,j]+betas[t+1,i]+self.calc_B(i, self.xs[t+1])+np.log(self.A[j,i,self.us[t]])

            zeta[t,:,:,:] -= sm.logsumexp(zeta[t, :,:,:])
            
            
        return gamma, zeta  
    
    
    def A_calc(self, gammas, zetas):
        
        A = np.zeros((self.N, self.N, self.M))
        
        zeta = sm.logsumexp(zetas, axis=-1)
        # A = sm.logsumexp(zeta[:,:,:,:], axis=0)
        # A -= sm.logsumexp(sm.logsumexp(zeta, axis=2), axis=0)[:,None,:]
           
        for u in range(self.M):
            for j in range(self.N): 
                for i in range(self.N):
                
                    A[j,i,u] = sm.logsumexp(zeta[:,j,i,u], axis=0)
                A[j,:,u] -= sm.logsumexp(zeta[:,j,:,u,])
        
        return np.exp(A)
                    
        
    def B_calc(self, gammas, zetas):
        
        if self.emission_distr == 'gaussian':
            
            gamma = sm.logsumexp(gammas, axis=-1)
            B = np.zeros((2, self.N, self.O2))
            
            # sum over gamma by time and data sequence
            B_denom = np.exp(sm.logsumexp(gamma, axis=0))
            
            gammas = np.exp(gammas)
            
            for o in range(self.O2):
                for j in range(self.N): 
                    
                    B_0 = gammas[:,j,:]*self.xs_all[:,o,:]
                    B_0 = np.sum(B_0, axis=-1)
                    
                    B_1 = gammas[:,j,:]*((self.xs_all[:,o,:] - self.B[0,j,o])**2)
                    B_1 = np.sum(B_1, axis=-1)
                    
                    B[0,j,o] = np.sum(B_0, axis=0)/ B_denom[j]
                    B[1,j,o] = np.sum(B_1, axis=0)/B_denom[j]
                    
        
        elif self.emission_distr == 'categorical':
            B = np.zeros((self.N, self.O))
            for x in range(self.O):
                mask = np.where(self.xs_all==x)
                B[:,x] = np.sum(np.sum(gammas[mask[0],:,mask[1]], axis=0), axis=-1)/np.sum(np.sum(gammas, axis=0), axis=-1)
                
        return B
    
    
    def EM(self):
        
        for it in range(self.num_iters):
            
            zetas = np.zeros((self.T-1, self.N, self.N,self.M, self.xs_all.shape[-1]))
            gammas = np.zeros((self.T, self.N, self.xs_all.shape[-1]))

            for d in range(self.xs_all.shape[-1]):
                
                self.xs = self.xs_all[...,d]
                if not self.hidden_inputs: self.us = self.us_all[:,d]

                gammas[:,:,d], zetas[:,:,:,:,d] = self.zeta_calc()
            
            
            # g = np.exp(gammas)
            # pi = np.sum(g[0,:,:],axis=-1)/self.xs_all.shape[-1]

            
            A = self.A_calc(gammas, zetas)
            B = self.B_calc(gammas, zetas)
            
            if (np.sum((A-self.A)**2) + np.sum((B-self.B)**2)) < 1e-4:
                print('Termintating at Iter', it)
                break
            else:
                if it%5==0:
                    print('Iteration', it)
                    print('Update Difference:' ,(np.sum((A-self.A)**2) + np.sum((B-self.B)**2)))
                self.A, self.B = A.copy(),B.copy()
               
        np.save('/u/kw5km/Research/Antibiotic_Resistance/Learned_Params/A_'+self.save+'_'+str(self.N),self.A)
        np.save('/u/kw5km/Research/Antibiotic_Resistance/Learned_Params/B_'+self.save+'_'+str(self.N),self.B)
        np.save('/u/kw5km/Research/Antibiotic_Resistance/Learned_Params/pi_'+self.save+'_'+str(self.N),self.pi)
            
        
        
    def predict(self, test, B_true=None):
        plt.switch_backend('agg')
        plt.close()
        
        repeats = self.xs_all.shape[-1]/(self.M**2 - (self.M-1))
        repeats = int(repeats)
            
        for o in range(self.O2):
            iters=100
            seen = [np.zeros((self.T,))+12]
            
            # print(test.shape)
            
            fig, ax = plt.subplots(2,1, figsize=(8, 8))
            
            for d in range(test.shape[2]):
                
                us, xs = test[:,0,d],test[:,1+o,d]
                if np.any(np.all(us==seen, axis=1)): continue
            
                seen.append(us)
                
                us_2 = np.ones_like(test[:,0,:])*us[:,None]
                idx = np.where((test[:,0,:]==us_2).all(axis=0))
                # print(idx[0])
                xs_all = test[:,1+o,idx[0]]
                xs_all = xs_all[:,:iters]
                # print(xs_all.shape)
                
                xs_mu = np.mean(xs_all, axis=1)
                xs_sigma = np.std(xs_all, axis=1)
                # print(xs_mu.shape, xs_sigma.shape)
                
                fig2, ax2 = plt.subplots(3,1, figsize=(9, 9))
                
                if repeats < iters:
                    ax2[1].plot(xs_all[:,repeats:], alpha=0.25)
                    ax2[1].plot(xs_all[:,:repeats], color='r', alpha=0.55)
                else: ax2[1].plot(xs_all, alpha=0.25)
                    
                
                pred = np.zeros((self.T, iters))
                
                # print('Treatments:', us)
            
                y_ts = np.zeros((self.T, iters))
                for it in range(iters):
                    
                    p_pi = self.pi/np.sum(self.pi)
                    y_prev = np.random.choice(self.N, 1, p=p_pi.flatten())
                    y_ts[0, it] = y_prev
                    for t in range(1,self.T):
                
                        treatment = int(us[t-1])
                        
                        # print('Pred Indices:', treatment, y_prev)
                        
                        p_A = self.A[y_prev,:,treatment]/np.sum(self.A[y_prev,:,treatment])
                        # print(p_A)/
                
                        y_t = np.random.choice(self.N, 1, p=p_A.flatten())
                        y_ts[t, it] = y_t
                        # y_t = np.argmax(self.A[y_prev,:,treatment])
                        
                        if self.emission_distr == 'categorical':
                            p_B = self.B[y_t,:]/np.sum(self.B[y_t,:])
                            x_t = np.random.choice(self.O, 1, p=p_B.flatten())
                            # x_t = np.argmax(self.B[y_t,:])#
                            
                        elif self.emission_distr == 'gaussian':
                            mu, sigma = self.B[0,y_t,o], np.sqrt(self.B[1,y_t,o])
                            x_t = st.norm.rvs(mu ,sigma)
                
                        pred[t, it] = x_t
                
                        y_prev = y_t
                    
                    
                pred_mu  = np.mean(pred, axis=1)
                pred_sigma  = np.std(pred, axis=1)
                
                # hs_mu  = np.mean(y_ts, axis=1)
                # hs_sigma  = np.std(y_ts, axis=1)
                
                pred_rmse = np.sqrt(np.mean((pred - xs_mu[:,None])**2, axis=0))
                min_rmse, max_rmse = np.min(pred_rmse), np.max(pred_rmse)
                
                ax2[0].errorbar(np.arange(self.T), pred_mu, pred_sigma,alpha=0.5, label='Pred', color='b')
                ax2[0].errorbar(np.arange(self.T), xs_mu, xs_sigma,alpha=0.5, label='True', color='r')
                # ax2[1].errorbar(np.arange(self.T), pred_mu, pred_sigma,alpha=0.5, color='b')
                ax2[2].plot(pred, alpha=0.1, color='b')
                
                ax2[1].set_ylim((np.min(self.xs_all)-1,np.max(self.xs_all)+1))
                ax2[1].set_yticks(np.arange(np.min(self.xs_all)-1,np.max(self.xs_all)+1,2))
                
                ax2[2].set_ylim((np.min(self.xs_all)-1,np.max(self.xs_all)+1))
                ax2[2].set_yticks(np.arange(np.min(self.xs_all)-1,np.max(self.xs_all)+1,2))
                
                ax2[0].set_ylim((np.min(self.xs_all)-1,np.max(self.xs_all)+1))
                ax2[0].set_yticks(np.arange(np.min(self.xs_all)-1,np.max(self.xs_all)+1,2))
                ax2[0].set_title('MIC for Treat. {} - Treatment: {} {}'.format(o+1,us[0], us[-1]) )
                ax2[0].annotate('Min RMSE: {0:.2f}, Max RMSE:{1:.2f}'.format(min_rmse, max_rmse), xy=(0.05, 0.9), xycoords='axes fraction')
                
                # ax2[2].errorbar(np.arange(self.T), hs_mu, hs_sigma,alpha=0.5, label='Hidden State')
                # ax2[2].set_ylim((0,self.N))
                # ax2[2].set_yticks(np.arange(self.N))
                
                ax2[0].legend(loc='upper right')
                
            if B_true is not None:
                for y in range(B_true.shape[0]):
                    ax[0].errorbar(np.arange(self.T), B_true[y,0,o]*np.ones((self.T,)), B_true[y,1,o]*np.ones((self.T,)),alpha=0.25, color='r')
                    
            for y in range(self.B.shape[1]):
                ax[0].errorbar(np.arange(self.T), self.B[0,y,o]*np.ones((self.T,)), np.sqrt(self.B[1,y,o])*np.ones((self.T,)),alpha=0.5, color='b')
                
            ax[1].hist(self.xs_all[...,o,:]) 
            ax[1].set_title('Hist. of MIC values in Data')
            
            ax[0].set_ylim((np.min(self.xs_all)-1,np.max(self.xs_all)+1))
            ax[0].set_yticks(np.arange(np.min(self.xs_all)-1,np.max(self.xs_all)+1,2))
            ax[0].set_title('Emission Params., MIC for Treat. {}'.format(o+1))

        
        fig4, ax4 = plt.subplots(nrows=self.M, ncols=2, figsize=(8, 12))
        ax4[0,1].set_title('Hidden State x Treat. Mean MIC')
        ax4[0,1].imshow(self.B[0,:,:], extent=[1,self.O2+1,self.N,0])
        
        for m in range(self.M):
            ax4[m,0].set_title('Transition Probs., Input {}'.format(m))
            ax4[m,0].imshow(self.A[:,:,m])
            
        for a in ax4[1:,1]:
            a.remove()
        
        pp = PdfPages('/u/kw5km/Research/Antibiotic_Resistance/Figs_Oct_21/'+self.save+'_'+str(self.N)+'.pdf')
        fgs = [plt.figure(n) for n in plt.get_fignums()]
        for fg in fgs:
            fg.savefig(pp, format='pdf')
        pp.close()
        # plt.show()
                       
