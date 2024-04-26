
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Tensorflow does not support numpy array operations for backprop
# def easy_PTM_depol_channel(depol_mat):
#     PTM_depol = (1 - np.abs(depol_mat)) * np.eye(4)
#     # Create a new tensor with the modified value at index [0, 0]
#     PTM_depol = np.array(PTM_depol)
#     PTM_depol[0, 0] = 1
#     PTM_depol = tf.convert_to_tensor(PTM_depol, dtype=tf.complex64)
#     return PTM_depol

def easy_PTM_depol_channel(depol_mat):
    identity = tf.eye(4, dtype=tf.float32)
    PTM_depol = (1 - tf.math.abs(depol_mat)) * identity
    
    # Update the value at index [0,0] to 1
    # We use Tensorflow operations to ensure gradient computation
    PTM_depol = tf.tensor_scatter_nd_update(PTM_depol, [[0,0]], [1])
    PTM_depol = tf.cast(PTM_depol, dtype=tf.complex64)
    return PTM_depol



def pauli_matrices():
    """Return the Pauli matrices including identity."""
    I = tf.eye(2, dtype=tf.complex64)
    X = tf.constant([[0, 1], [1, 0]], dtype=tf.complex64)
    Y_imag = tf.constant([[0, -1], [1, 0]], dtype=tf.float32)
    Y_real = tf.constant([[0, 0], [0, 0]], dtype=tf.float32)
    Y = tf.complex(Y_real, Y_imag)
    # Y = tf.constant([[0, -1j], [1j, 0]], dtype=tf.complex64)
    Z = tf.constant([[1, 0], [0, -1]], dtype=tf.complex64)
    
    return [I, X, Y, Z]

def compute_ideal_ptm(unitary):
    """Compute the ideal PTM from a given unitary."""
    paulis = pauli_matrices()
    ptm_ideal = tf.zeros((4, 4), dtype=tf.complex64)

    for i in range(4):
        for j in range(4):
            term = tf.matmul(unitary, tf.matmul(paulis[j], tf.linalg.adjoint(unitary)))
            trace_value = 0.5 * tf.linalg.trace(tf.matmul(paulis[i], term))
            
            # Update ptm_ideal at position [i, j] with the calculated trace_value
            indices = tf.constant([[i, j]])
            ptm_ideal = tf.tensor_scatter_nd_add(ptm_ideal, indices, [trace_value])
            
    return ptm_ideal


def general_custom_gate(theta, delta, depol_amt, gate):
    # Compute real and imaginary parts as real numbers initially
    real_part = tf.cos((theta + delta) / 2)
    imag_part = tf.sin((theta + delta) / 2)
    
    # Cast them to complex numbers only when necessary
    unitary_rx_adjusted = tf.cast(real_part, dtype=tf.complex64) * tf.eye(2, dtype=tf.complex64) - 1j * tf.cast(imag_part, dtype=tf.complex64) * pauli_matrices()[gate]
    
    ptm_adjusted_rx = compute_ideal_ptm(unitary_rx_adjusted)
    ptm = tf.matmul(easy_PTM_depol_channel(depol_amt), ptm_adjusted_rx)
    
    return tf.math.real(ptm)

# X-gate
X_Ideal = general_custom_gate(theta=math.pi/2, delta=0.0, depol_amt=0.00, gate=1)
X_QC = general_custom_gate(theta=math.pi/2, delta=0.1, depol_amt=0.01, gate=1)
X_GST = general_custom_gate(theta=math.pi/2, delta=0.100261497, depol_amt=0.010365579, gate=1)

# Y-gate
Y_Ideal = general_custom_gate(theta=math.pi/2, delta=0.0, depol_amt=0.00, gate=2)
Y_QC = general_custom_gate(theta=math.pi/2, delta=0.2, depol_amt=0.02, gate=2)
Y_GST = general_custom_gate(theta=math.pi/2, delta=0.199429939, depol_amt=0.020106427, gate=2)

# How bad is the QC, or how much is our knowledge gap, or how much is there to learn from the GST
X_diff_QC_Ideal = np.array(X_QC - X_Ideal)
Y_diff_QC_Ideal = np.array(Y_QC - Y_Ideal)

# How much was the GST not able to learn
X_diff_GST_QC = np.array(X_GST - X_QC)
Y_diff_GST_QC = np.array(Y_GST - Y_QC)

# Plotting the heatmaps

max_abs_ideal = 0.2 # np.max([np.abs(X_diff_QC_Ideal),np.abs(Y_diff_QC_Ideal)])
max_abs_ml4qgst = 0.0006 # np.max([np.abs(X_diff_GST_QC),np.abs(Y_diff_GST_QC)])

fig, ax = plt.subplots(2,2)

fs = 20
ls = 14

ax[0][0].set_title('a) X Gate: (QC - Ideal)',size=fs)
sp1 = ax[0][0].matshow(X_diff_QC_Ideal, cmap='viridis', vmin=-max_abs_ideal, vmax=max_abs_ideal) # coolwarm
divider = make_axes_locatable(ax[0][0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cb1 = fig.colorbar(sp1, cax=cax, orientation='vertical')
cb1.ax.tick_params(labelsize=ls)

ax[0][1].set_title('b) X Gate: (ML4QGST - QC)',size=fs)
sp2 = ax[0][1].matshow(X_diff_GST_QC, cmap='seismic', vmin=-max_abs_ml4qgst, vmax=max_abs_ml4qgst) # coolwarm
divider = make_axes_locatable(ax[0][1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cb2 = fig.colorbar(sp2, cax=cax, orientation='vertical')
cb2.ax.tick_params(labelsize=ls)

ax[1][0].set_title('c) Y Gate: (QC - Ideal)',size=fs)
sp3 = ax[1][0].matshow(Y_diff_QC_Ideal, cmap='viridis', vmin=-max_abs_ideal, vmax=max_abs_ideal) # coolwarm
divider = make_axes_locatable(ax[1][0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cb3 = fig.colorbar(sp3, cax=cax, orientation='vertical')
cb3.ax.tick_params(labelsize=ls)

ax[1][1].set_title('d) Y Gate: (ML4QGST - QC)',size=fs)
sp4 = ax[1][1].matshow(Y_diff_GST_QC, cmap='seismic', vmin=-max_abs_ml4qgst, vmax=max_abs_ml4qgst) # coolwarm
divider = make_axes_locatable(ax[1][1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cb4 = fig.colorbar(sp4, cax=cax, orientation='vertical')
cb4.ax.tick_params(labelsize=ls)

ax[0][0].set_xticks(np.arange(4))
ax[0][0].set_yticks(np.arange(4))
ax[0][0].set_xticklabels(np.arange(1, 4+1), fontsize=ls)
ax[0][0].set_yticklabels(np.arange(1, 4+1), fontsize=ls)
ax[0][0].xaxis.set_ticks_position('bottom')

ax[0][1].set_xticks(np.arange(4))
ax[0][1].set_yticks(np.arange(4))
ax[0][1].set_xticklabels(np.arange(1, 4+1), fontsize=ls)
ax[0][1].set_yticklabels(np.arange(1, 4+1), fontsize=ls)
ax[0][1].xaxis.set_ticks_position('bottom')

ax[1][0].set_xticks(np.arange(4))
ax[1][0].set_yticks(np.arange(4))
ax[1][0].set_xticklabels(np.arange(1, 4+1), fontsize=ls)
ax[1][0].set_yticklabels(np.arange(1, 4+1), fontsize=ls)
ax[1][0].xaxis.set_ticks_position('bottom')

ax[1][1].set_xticks(np.arange(4))
ax[1][1].set_yticks(np.arange(4))
ax[1][1].set_xticklabels(np.arange(1, 4+1), fontsize=ls)
ax[1][1].set_yticklabels(np.arange(1, 4+1), fontsize=ls)
ax[1][1].xaxis.set_ticks_position('bottom')

fig.set_size_inches(12, 10)
plt.subplots_adjust(left=0.0, right=0.98, top=0.95, bottom=0.06, wspace=0.0, hspace=0.23)

plt.savefig('heatmaps.eps', format='eps', dpi=300)
plt.show()