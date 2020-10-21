from __future__ import division
import numpy as np
import math
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def init_genes(length):
    genes = np.zeros((length,))
    return genes

def calc_MIC(genes, weights):
    MICs = []
    for g in genes:
        MIC = g@weights
        MIC = np.log2(64/(1+50*math.exp(-MIC))) + st.norm.rvs(0, .1)
        
        MICs.append(MIC)
    return MICs

    # total_MIC = np.sum(weights[weights>0])
    # MIC = 1
    # for g in range(genes.size):
    #     # print(MIC, weights[g])
    #     if genes[g] != 1: continue
    #     MIC += weights[g]*total_MIC
    # return MIC

    # MIC = genes@weights
    # if MIC <= 0: 
    #     MIC= 1/(math.exp(2))
    # return MIC

def run_generations(genes, treatments, generations, probs, weights, prev=None, reversion_rates=None):
    m_gs=[]
    mutated_genes = genes
    for treatment, generation in zip(treatments, generations):
        w = weights[treatment,:].copy()
        mutation_probs = probs[treatment,:].copy()

        if (prev==None) or (prev==treatment): 
            prev_w = np.zeros_like(w)
            mutation_probs_prev=np.zeros_like(mutation_probs)
        else:

            prev_w = weights[prev, :].copy()
            prev_w[prev_w>0] = 1
            prev_w[prev_w<0] = 0

            mutation_probs_prev = np.zeros_like(mutation_probs)
            
            rate = reversion_rates[prev]
            prob_mask = np.random.rand(prev_w.size)
            idx_revert = np.logical_and(prev_w>0, prob_mask<rate)

            # print('Current Treatment:', treatment, ', Prev Treatment:', prev, ', Rate:', rate)

            mutation_probs_prev[idx_revert] = 2e-5*((50-100) * np.random.random_sample(prev_w[idx_revert].sum(dtype=int)) + 100)

        for gen in range(generation):
            # print(mutated_genes)

            rndm_rolls = np.random.rand(genes.size)
            # print(rndm_rolls, mutation_probs)
            mutate = np.logical_or(rndm_rolls<mutation_probs, rndm_rolls<mutation_probs_prev)

            idx_ = np.logical_and(w>0, mutated_genes==1)
            idx_prev = np.logical_and(prev_w>0, mutated_genes==0)
            idx = np.logical_or(idx_, idx_prev)

            mutate[idx] = 0
            # print(mutate)
            # print(np.sum(mutate))
            mutated_genes = (mutated_genes + mutate)%2
            
            # print('Treatement: {}, Gen: {}, Genes: {}'.format(treatment , gen, mutated_genes))
            if gen%72==0: m_gs.append(mutated_genes)
    return m_gs


def main():
    np.random.seed(11)
    genes = init_genes(100)

    treatments = np.arange(3)

    weights = np.zeros((treatments.size, genes.size))

    # e. coli base mutation rate
    # base_rate = 2e-6
    base_rate = 5e-6

    # Treatment 0 is always control 
    probs = base_rate*np.ones((treatments.size, genes.size))

    reversion_rates = ((.55-1) * np.random.rand(treatments.size) + 1)
    # reversion_rates = ((.15-.4) * np.random.rand(treatments.size) + .4)
    
    high_aff = np.zeros_like(genes)

    for i in range(1,treatments.size): #change to range(1, tr.size) if including control
        
        #TEST:
        # num_affected = np.random.randint(8,10)
        # if i==1:
        # num_affected = 2
        # else: num_affected = 1
        num_affected = np.random.randint(genes.size*(0.025),genes.size*(0.1))
        print(i, num_affected)
        # num_affected = np.random.randint(genes.size*(0.02),genes.size*(0.04))
    

        highly_affected = np.zeros(probs[i,:].shape, bool)

        chosen = np.random.choice(range(genes.size-50),size =(num_affected,), replace=False)
        highly_affected[chosen] = 1
        # print('Treat {}, Highly Affected: {}'.format(i, highly_affected))

        high_aff= np.logical_or(high_aff, highly_affected)

        # print('highly affected: ',highly_affected)

        # 2 - 4x the base rate of mutation
        # highly_affected_probs = base_rate*((50-75) * np.random.random_sample(num_affected) + 75)
        highly_affected_probs = base_rate*((100-150) * np.random.random_sample(num_affected) + 150)
        print(highly_affected_probs)

        probs[i, highly_affected] = highly_affected_probs
        # probs[i, ~highly_affected] = base_rate

        # weights[i, highly_affected] = 3e3*(probs[i,highly_affected])
        weights[i, highly_affected] = 3e3*(probs[i,highly_affected])

        # Roll for negative affect:
        if np.random.rand() > 1:#0.6:

            # Genes that affect other ABR negatively
            treatmnts = np.arange(1,treatments.size)
            current = np.where(treatmnts==i)[0]
            treatmnts = np.delete(treatmnts, current)

            num_neg = 1# np.random.randint(1,5)
            # num_neg = np.random.randint(0.05*num_affected, 0.10*num_affected)
            
            # treat_neg = np.random.choice(treatmnts,size =(1,), replace=False)
            treat_neg = np.random.choice(treatmnts)*np.ones((num_neg,), dtype=int)# np.random.choice(treatmnts,size =(num_neg,), replace=True)
            
            neg_idx = np.random.choice(chosen,size =(num_neg,), replace=False)

            
            weights[treat_neg, neg_idx] = -9e2*(probs[i,neg_idx])
            # weights[treat_neg,neg_idx] = -9e2*(probs[i,neg_idx])
            
            print('Treatment {} Affects Treatment {} Neg. at {}'.format(i, treat_neg, neg_idx))
    
    total_days = [20, 20]
    days = np.arange(total_days[0]+total_days[1])
    
    repeats = 100
    plot_flag = 0
    
    n = high_aff.sum()
    # print(high_aff)
    print('Num. Affected:', n)
    
    # gene_permutations = np.array(np.indices(n * (2,))).reshape(n, -1)
    # gene_permutations[:, np.argsort(gene_permutations.sum(0)[::-1], kind='mergesort')].T[::-1]
    
    # mutation_count = np.zeros((gene_permutations.shape[1]))
    # std = np.zeros((gene_permutations.shape[1], total_days[0]+total_days[1], treatments.size+(treatments.size-1)**2, repeats, treatments.size-1))
                              
    

    data_treat = np.zeros((len(days), treatments.size+(treatments.size-1)**2, repeats))
    data_MIC = np.zeros((len(days),treatments.size-1, treatments.size+(treatments.size-1)**2, repeats)) 

    for repeat in range(repeats):
        
        if repeat%10==0: print(repeat)
        
        if plot_flag:
            plt.rc('text', usetex=False)
            plt.rc('font', family='serif')
            
            fig, ax = plt.subplots(treatments.size-1, treatments.size-1, figsize=(8, 8))
    
            fig.text(0.5, 0.04, 'Day', ha='center', va='center')
            fig.text(0.05, 0.5, '$\log_2$(MIC)', ha='center', va='center', rotation='vertical')
            
            
            for m in range(treatments.size-1):
                ax[m,0].set_ylabel('Treatment {} MIC'.format(m+1))
            
                ax[0,m].set_title('Treatment {} Lineages'.format(m+1))
    
            colors = ['k','r', 'b', 'y']

        t_idx=0

        for treat1 in range(0,treatments.size):
            mutated_genes = [genes.copy()]
            MICs = np.zeros((treatments.size-1,len(days)))
            

            treatment, generation = [treat1], [72*(total_days[0])]
            mutated_genes = run_generations(mutated_genes[-1], treatment, generation, probs, weights)
            
            for m in range(treatments.size-1):
                MICs[m,:total_days[0]] = calc_MIC(mutated_genes, weights[m+1,:])
                # MICs[m,0] = np.max((0.1, MICs[m,0] + st.norm.rvs(0, .5)))
            
            # for day in range(total_days[0]):
                
            #     g = mutated_genes[day]
                
            #     g2 = np.ones_like(gene_permutations)*g[high_aff,None]
            #     gene_idx = np.where((gene_permutations==g2).all(axis=0))
                
            #     mutation_count[gene_idx]+=1

            #     std[gene_idx, day, t_idx, repeat,:] = np.log2(MICs[:,day])


            for treat2 in range(0,treatments.size):
                mutated_genes_t2 = mutated_genes.copy()
                if treat1==0 and treat2!=0:continue

                # for day in range(21,40):
                treatment, generation = [treat2], [72*total_days[1]]
                
                mutated_genes_t2 = run_generations(mutated_genes_t2[-1], treatment, generation, probs, weights, prev=treat1, reversion_rates=reversion_rates)
                
                for m in range(treatments.size-1):
                    MICs[m,total_days[0]:] = calc_MIC(mutated_genes_t2, weights[m+1,:])                    
                
                # for day in range(total_days[1]):
                        
                #     g = mutated_genes_t2[day]
                    
                #     g2 = np.ones_like(gene_permutations)*g[high_aff,None]
                #     gene_idx = np.where((gene_permutations==g2).all(axis=0))
                    
                #     mutation_count[gene_idx]+=1
                
                #     std[gene_idx, day+total_days[0], t_idx, repeat,:] = np.log2(MICs[:,day+total_days[0]])

                data_MIC[:total_days[0],:,t_idx,repeat] = MICs[:,:total_days[0]].T
                data_MIC[total_days[0]:,:,t_idx,repeat] = MICs[:,total_days[0]:].T

                data_treat[:total_days[0],t_idx, repeat] = treat1
                data_treat[total_days[0]:,t_idx, repeat] = treat2
                t_idx +=1
                
                if plot_flag:
                    if treat1==0 and treat2==0: 
                        # PLOT ON ALL AX
                        for r in range(treatments.size-1):
                            for c in range(treatments.size-1):
                                ax[r,c].plot(days, MICs[r,:],color=colors[treat1], alpha=0.75)
                                ax[r,c].set_ylim((-4,12))
                                ax[r,c].set_yticks(np.arange(-4,12,2))
                    else: 
                        # PLOT ON AX=(treat1-1, range(0,3))
                        for r in range(treatments.size-1):

                            ax[r,treat1-1].plot(days[:total_days[0]+1], MICs[r,:total_days[0]+1],color=colors[treat1], alpha=0.75)
                            ax[r,treat1-1].plot(days[total_days[0]:], MICs[r,total_days[0]:],color=colors[treat2],linestyle='--', alpha=0.75) 
        
        
        # # Create the legend
        if plot_flag:
            custom_lines = [Line2D([0], [0], color=colors[0], lw=1),
                        Line2D([0], [0], color=colors[1], lw=1),
                        Line2D([0], [0], color=colors[2], lw=1),
                        Line2D([0], [0], color=colors[3], lw=1)]
            fig.legend(custom_lines, ['Control', 'Treatment 1', 'Treatement 2', 'Treatment 3'], loc='upper right')
    
            plt.show() 
    
    
    weights_counts = weights.copy()
    weights_counts[weights_counts<0]=0
    weights_counts[weights_counts>0]=1
    print('1,2 Shared Genes: ', np.sum(np.logical_and(weights_counts[1,:]==weights_counts[2,:],weights_counts[1,:]==1)))
    # print('1,3 Shared Genes: ', np.sum(np.logical_and(weights_counts[1,:]==weights_counts[3,:],weights_counts[1,:]==1)))
    # print('2,3 Shared Genes: ', np.sum(np.logical_and(weights_counts[3,:]==weights_counts[2,:],weights_counts[3,:]==1)))
    
    
    # std[std==0] = np.nan        
    # mean_fin = np.nanmean(std,axis=(1,2,3))
    # std_fin = np.nanstd(std,axis=(1,2,3))

    # B_true = np.stack((mean_fin,std_fin), axis=1)

    
    # for tr in range(treatments.size-1):
    #     fig2, ax2 = plt.subplots(1,1, figsize=(8, 8))
    #     for y in range(B_true.shape[0]):
    #         ax2.errorbar(np.arange(total_days[0]+total_days[1]), B_true[y,0,tr]*np.ones((total_days[0]+total_days[1],)), B_true[y,1,tr]*np.ones((total_days[0]+total_days[1],)),alpha=0.25, color='r')
    #     ax2.set_ylim((-4,12))
    #     ax2.set_yticks(np.arange(-4,12,2))

    plt.show()    
        
    if plot_flag==0:
        np.save('../Sim_Data/MIC_256hs',data_MIC)
        np.save('../Sim_Data/treat_256hs',data_treat)
        # np.save('../Processed_Data/Sim_Data_means_all_small_2',mean_fin)
        # np.save('../Processed_Data/Sim_Data_stds_all_small_2',std_fin)

if __name__== "__main__":
  main()