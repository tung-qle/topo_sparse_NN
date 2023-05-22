from re import M
import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np

def plot_compare(normfc1, normfc2, normfc, emp_loss, true_loss, err_normfc1, err_normfc2, err_emp_loss, err_true_loss, err_normfc, label, alpha = 0.5, sum_weight = True, fontsize = 15):
    plt.figure(figsize=(19,5))
    plt.title("Training of non-closed fixed support neural network", fontsize = fontsize)
    x = np.arange(1, len(normfc1[0]) + 1)
    ax1 = plt.subplot(1,3,1)
    ax1.plot(x, emp_loss[0], label=label[0], color = "blue", linewidth=0.5)
    ax1.fill_between(x, emp_loss[0] - err_emp_loss[0], emp_loss[0]+ err_emp_loss[0], color = 'blue', alpha = alpha)
    ax1.plot(emp_loss[1], label=label[1], color = "green", linewidth=0.5)
    ax1.fill_between(x, emp_loss[1] - err_emp_loss[1], emp_loss[1]+ err_emp_loss[1], color = 'green', alpha = alpha)
    ax1.set_title('a)', fontsize = 15)

    ax1.set_xlabel('Iterations', fontsize = fontsize)
    ax1.set_ylabel('Emperical loss', fontsize = fontsize)
    ax1.legend(prop={'size': 13})

    ax2 = plt.subplot(1,3,2)
    ax2.plot(x, true_loss[0], label=label[0], color = "blue")
    ax2.fill_between(x, true_loss[0] - err_true_loss[0], true_loss[0] + err_true_loss[0], color = 'blue', alpha = alpha)
    ax2.plot(x, true_loss[1], label=label[1], color = "green")
    ax2.fill_between(x, true_loss[1] - err_true_loss[1], true_loss[1] + err_true_loss[1], color = 'green', alpha = alpha)    
    ax2.set_xlabel('Iterations', fontsize = fontsize)
    ax2.set_ylabel('True loss', fontsize = fontsize)
    ax2.set_title('b)', fontsize = fontsize)
    ax2.legend(prop={'size': 13})

    ax3 = plt.subplot(1,3,3)
    if sum_weight:
        ax3.plot(x, normfc[0], label = label[0], color = 'blue')
        ax3.fill_between(x, normfc[0] - err_normfc[0], normfc[0] + err_normfc[0], color = 'blue', alpha = alpha)
        ax3.plot(x, normfc[1], label = label[1], color = 'green')
        ax3.fill_between(x, normfc[1] - err_normfc[1], normfc[1] + err_normfc[1], color = 'green', alpha = alpha)
    else: 
        ax3.plot(x, normfc1[0], label = label[0], color = 'blue')
        ax3.fill_between(x, normfc1[0] - err_normfc1[0], normfc1[0] + err_normfc1[0], color = 'blue', alpha = alpha)    
        ax3.plot(x, normfc2[0], color = 'blue')
        ax3.fill_between(x, normfc2[0] - err_normfc2[0], normfc2[0] + err_normfc2[0], color = 'blue', alpha = alpha)    
        ax3.plot(x, normfc1[1], label = label[1], color = 'green')
        ax3.fill_between(x, normfc1[1] - err_normfc1[1], normfc1[1] + err_normfc1[1], color = 'green', alpha = alpha)    
        ax3.plot(x, normfc2[1], color = 'green')
        ax3.fill_between(x, normfc2[1] - err_normfc2[1], normfc2[1] + err_normfc2[1], color = 'green', alpha = alpha)    
    ax3.set_xlabel('Iterations', fontsize = fontsize)
    ax3.set_ylabel('Weight matrix norm', fontsize = fontsize)
    ax3.set_title('c)', fontsize = fontsize)
    ax3.legend(loc = 'upper left', prop={'size': 13})

    #plt.suptitle("Evolution during training of losses and weight matrices norm", fontsize = fontsize)
    plt.savefig(fname='training_behavior_compare.png', dpi = 200)
    plt.show()

# result = np.load("training_evo_full.npz")
# normfc1 = result['arr_0']
# normfc2 = result['arr_1']
# emp_loss = result['arr_2']
# true_loss = result['arr_3']

# plot(normfc1, normfc2, emp_loss, true_loss)
if __name__ == "__main__":
    d = 100
    with open('training_evo_LU.pickle', 'rb') as handle:
        dict = cPickle.load(handle)
    handle.close()

    with open("training_evo_regularisation_LU.pickle", 'rb') as handle:
        dict_reg = cPickle.load(handle)
    handle.close()

    norm_noreg = np.sqrt(np.square(np.array(dict['normfc1'])) + np.square(np.array(dict['normfc2']))) 
    norm_reg = np.sqrt(np.square(np.array(dict_reg['normfc1'])) + np.square(np.array(dict_reg['normfc2']))) 
    # Calculate mean of norm
    m_norm1_noreg = np.mean(np.array(dict['normfc1']), axis = 0) 
    m_norm2_noreg = np.mean(np.array(dict['normfc2']), axis = 0) 
    m_norm1_reg = np.mean(np.array(dict_reg['normfc1']), axis = 0) 
    m_norm2_reg = np.mean(np.array(dict_reg['normfc2']), axis = 0) 
    m_norm_reg = np.mean(norm_reg, axis = 0) 
    m_norm_noreg = np.mean(norm_noreg, axis = 0) 

    # Calculate std of norm
    v_norm1_noreg = np.std(np.array(dict['normfc1']), axis = 0) 
    v_norm2_noreg = np.std(np.array(dict['normfc2']), axis = 0) 
    v_norm_noreg = np.std(norm_noreg, axis = 0) 
    v_norm1_reg = np.std(np.array(dict_reg['normfc1']), axis = 0) 
    v_norm2_reg = np.std(np.array(dict_reg['normfc2']), axis = 0) 
    v_norm_reg = np.std(norm_reg, axis = 0) 

    # Calculate mean of empirical loss
    m_em_loss_noreg = np.mean(np.array(dict['emp_loss']), axis = 0)
    m_em_loss_reg = np.mean(np.array(dict_reg['emp_loss']), axis = 0) 

    # Calculate std of empirical loss
    v_em_loss_noreg = np.std(np.array(dict['emp_loss']), axis = 0)
    v_em_loss_reg = np.std(np.array(dict_reg['emp_loss']), axis = 0)

    # Calculate mean of true loss
    m_true_loss_noreg = np.mean(np.array(dict['true_loss']), axis = 0)
    m_true_loss_reg = np.mean(np.array(dict_reg['true_loss']), axis = 0)

    # Calculate std of true loss
    v_true_loss_noreg = np.std(np.array(dict['true_loss']), axis = 0)
    v_true_loss_reg = np.std(np.array(dict_reg['true_loss']), axis = 0)

    normfc1 = [m_norm1_noreg, m_norm1_reg]
    normfc2 = [m_norm2_noreg, m_norm2_reg]
    normfc = [m_norm_noreg, m_norm_reg]
    emp_loss = [m_em_loss_noreg, m_em_loss_reg]
    true_loss = [m_true_loss_noreg, m_true_loss_reg]
    err_normfc1 = [v_norm1_noreg, v_norm1_reg]
    err_normfc2 = [v_norm2_noreg, v_norm2_reg]
    err_emp_loss = [v_em_loss_noreg, v_em_loss_reg]
    err_true_loss = [v_true_loss_noreg, v_true_loss_reg]
    err_normfc = [v_norm_noreg, v_norm_reg]
    label = ["No l2 regularisation", "l2 regularisation"]

    plot_compare(normfc1, normfc2, normfc, emp_loss, true_loss, err_normfc1, err_normfc2, err_emp_loss, err_true_loss, err_normfc, label)