# Graph Neural Networks and Applied Linear Algebra
# 
# Copyright 2023 National Technology and Engineering Solutions of
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with
# NTESS, the U.S. Government retains certain rights in this software. 
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer. 
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution. 
# 
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
# 
# 
# 
# For questions, comments or contributions contact 
# Chris Siefert, csiefer@sandia.gov 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import getSmallBandMatrices as gsbm

# plt.rcParams['text.usetex'] = True

# Save command:
# np.savez('test_eigenvalues', 
#          evals_A=evals_A, 
#          evals_DinvA=evals_DinvA, 
#          evals_TwoThirds_DinvA=evals_TwoThirds_DinvA,
#          evals_opt_DinvA=evals_opt_DinvA,
#          evals_learn_DinvA=evals_learn_DinvA,
#          diag_A=diag_A,
#          diag_opt_Dinv=diag_opt_Dinv,
#          diag_learn_Dinv=diag_learn_Dinv,
#          hs=hs,
#          band_locs=band_locs)

data = np.load('test_eigenvalues.npz')

evals_A = data['evals_A']
evals_DinvA = data['evals_DinvA']
evals_TwoThirds_DinvA = data['evals_TwoThirds_DinvA']
evals_opt_DinvA = data['evals_opt_DinvA']
evals_learn_DinvA = data['evals_learn_DinvA']
diag_A = data['diag_A']
diag_opt_Dinv = data['diag_opt_Dinv']
diag_learn_Dinv = data['diag_learn_Dinv']
h = data['hs']
band_loc = data['band_locs']

max_evals_A = np.max(evals_A, axis=1)
max_evals_DinvA = np.max(evals_DinvA, axis=1)
max_evals_TwoThirds_DinvA = np.max(evals_TwoThirds_DinvA, axis=1)
max_evals_opt_DinvA = np.max(evals_opt_DinvA, axis=1)
max_evals_learn_DinvA = np.max(evals_learn_DinvA, axis=1)

# histogram parameters
n_bins = 30

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def eigenvalue_scatter_plots(save=False):
    examples = [3, 5]

    # for i in range(evals_opt_DinvA.shape[0]):
    for i in examples:
        range_to_plot = evals_DinvA.shape[1]-10
        plt.scatter(range(range_to_plot, evals_DinvA.shape[1]), evals_DinvA[i, range_to_plot:], label='$\omega = 1$')
        plt.scatter(range(range_to_plot, evals_TwoThirds_DinvA.shape[1]), evals_TwoThirds_DinvA[i, range_to_plot:], label = '$\omega = 2/3$')
        plt.scatter(range(range_to_plot, evals_opt_DinvA.shape[1]), evals_opt_DinvA[i, range_to_plot:], label='$\omega_{co}$')
        plt.scatter(range(range_to_plot, evals_learn_DinvA.shape[1]), evals_learn_DinvA[i, range_to_plot:], label='Learned Diagonal')
        plt.legend(loc='upper left')
        ax = plt.gca()
        ax.set_xticks([])
        if save:
            plt.savefig(f'Sample_{i}')
            plt.close()
        else:
            plt.show()


# Separate histograms
def seperate_histograms(save=False):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.hist(max_evals_DinvA, bins=n_bins, label='Dinv A')
    ax2.hist(max_evals_TwoThirds_DinvA, bins=n_bins, label = '2/3 Dinv A')
    ax3.hist(max_evals_opt_DinvA, bins=n_bins, label='opt Dinv A')
    ax4.hist(max_evals_learn_DinvA, bins=n_bins, label='learned Dinv A')
    ax1.set_title('Dinv A')
    ax2.set_title('2/3 Dinv A')
    ax3.set_title('opt Dinv A')
    ax4.set_title('learned Dinv A')
    if save:
        raise NotImplementedError('save is not implemented for separate_histograms')
    else:
        plt.show()


# Stacked histograms
def stacked_histograms(save=False):
    max_eigvals = np.vstack((max_evals_DinvA, max_evals_TwoThirds_DinvA, max_evals_opt_DinvA, max_evals_learn_DinvA)).T
    labels = ['Dinv A', '2/3 Dinv A', 'opt Dinv A', 'learned Dinv A']
    plt.hist(max_eigvals, bins=n_bins, stacked=True, label=labels)
    plt.legend()
    if save:
        raise NotImplementedError('save is not implemented for stacked_histograms')
    else:
        plt.show()

# Stacked box-and-whisker
def box_and_whisker(save=False):
    max_eigvals = np.vstack((max_evals_DinvA, max_evals_TwoThirds_DinvA, max_evals_opt_DinvA, max_evals_learn_DinvA)).T
    plt.boxplot(max_eigvals)
    if save:
        raise NotImplementedError('save is not implemented for stacked_histograms')
    else:
        plt.show()

# Plot standard options with learned as the reference
def histograms_compared_to_learned(save=False):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    fig.suptitle('Comparison of maximum eigenvalues of $I - \omega V_{hf}^T D^{-1}AV_{hf}$')
    max_ref_eigs = max_evals_DinvA - max_evals_learn_DinvA
    ax1.hist(max_ref_eigs, bins=n_bins)
    ax1.axvline(0.0, color='k', linestyle='dashed', linewidth=1)
    ax1.set_title('Learned D vs $\omega = 1$')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('$\lambda_{learned} - \lambda_{\omega=1}$')
    max_ref_eigs = max_evals_TwoThirds_DinvA - max_evals_learn_DinvA
    ax2.hist(max_ref_eigs, bins=n_bins)
    ax2.axvline(0.0, color='k', linestyle='dashed', linewidth=1)
    ax2.set_title('Learned D vs $\omega = 2/3$')
    ax2.set_xlabel('$\lambda_{learned} - \lambda_{\omega=2/3}$')
    max_ref_eigs = max_evals_opt_DinvA - max_evals_learn_DinvA
    ax3.hist(max_ref_eigs, bins=n_bins)
    ax3.axvline(0.0, color='k', linestyle='dashed', linewidth=1)
    ax3.set_title('Learned D vs $\omega_{co}$')
    ax3.set_xlabel('$\lambda_{learned} - \lambda_{\omega_{co}}$')
    if save:
        plt.savefig('JacobiReferenceHists.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Winners plot
def winners_plot(save=False):
    all_maxes = np.vstack((max_evals_DinvA, max_evals_TwoThirds_DinvA, max_evals_opt_DinvA, max_evals_learn_DinvA)).T
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[0:4]
    colors[0] = colors[2]
    print(np.argmin(all_maxes, axis=1))
    plt.scatter(h, band_loc, c=np.argmin(all_maxes, axis=1), cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('band width')
    plt.ylabel('band location')
    if save:
        plt.savefig('JacobiWinners.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Plot an example digaonal across a horizontal line (similar to Illinois reference)
def example_diag_horizontal(save=False):
    # Find matrices with a middle-ish band location and smaller band width
    poss_idx = np.where((band_loc < 0.7)*(band_loc > 0.3)*(h < 0.004) == True)[0]

    # This one is a matrix in the lower right corner of the winners plot where the model wins as well
    # poss_idx = np.where((band_loc < 0.2)*(h > 0.013) == True)[0]

    # Take the first one
    ex_idx = poss_idx[0]

    M = diag_A[ex_idx].shape[0]

    Nx = int(1 + np.sqrt(1 + M))
    Ny = int(M / Nx)

    def idx_to_color(idx):
        # This is a little bit of a hack and depends on there being only 4 unique diagonals
        if diag_A[ex_idx, idx] == np.unique(diag_A[ex_idx])[2]:
            return 2         
        # This is a little bit of a hack and depends on there being only 4 unique diagonals
        if diag_A[ex_idx, idx] == np.unique(diag_A[ex_idx])[3]:
            return 3         
        if (idx < Nx) or (idx % Nx == 0) or (idx % Nx == Nx - 1) or (idx > (Ny-1)*Nx):
            return 1
        return 0

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = colors[0:4]
    cs = [idx_to_color(i) for i in range(diag_A[ex_idx].shape[0])]
    my_cmap = matplotlib.colors.ListedColormap(colors)
    classes = ['interior', 'boundary', 'band edge', 'band center']

    # Plot the diagonal
    fig, ax = plt.subplots(1,1)
    scatter = ax.scatter(range(diag_learn_Dinv[ex_idx].shape[0]), diag_learn_Dinv[ex_idx], s=10, c=cs, cmap=my_cmap)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    if save:
        plt.savefig('ExampleDiagonal_scatter.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def example_diag_surface(save=False):
    # Find matrices with a middle-ish band location and smaller band width
    poss_idx = np.where((band_loc < 0.7)*(band_loc > 0.3)*(h < 0.004) == True)[0]

    # This one is a matrix in the lower right corner of the winners plot where the model wins as well
    # poss_idx = np.where((band_loc < 0.2)*(h > 0.013) == True)[0]

    # Take the first one
    ex_idx = poss_idx[0]

    print(f'example_diag_surface: h = {h[ex_idx]}, band_loc = {band_loc[ex_idx]}')

    # Plot the diagonal
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # get the xy vector, this is a bit of a hack, we regenerate the matrix
    M = diag_A[ex_idx].shape[0]
    Nx = int(1 + np.sqrt(1 + M))
    Ny = int(M / Nx)
    _, xy, _ = gsbm.getSmallBandMatrix(Nx, h[ex_idx], band_loc[ex_idx],False)

    X = xy[:,0].reshape(-1, Nx)
    Y = xy[:,1].reshape(-1, Nx)

    new_cmap = truncate_colormap(matplotlib.colormaps['Blues'], 0.3, 1.0)

    vmin = min(np.min(diag_opt_Dinv), np.min(diag_learn_Dinv))
    vmax = max(np.max(diag_opt_Dinv), np.max(diag_learn_Dinv))

    surf = ax.plot_surface(X, Y, diag_learn_Dinv[ex_idx].reshape(Ny,Nx), vmin=vmin, vmax=vmax, cmap=new_cmap)

    fig.set_figheight(10)
    fig.set_figwidth(12)
    fig.colorbar(surf, shrink=0.8)
    plt.title('Learned Diagonal for h = 0.003, band location = 0.405')
    if save:
        plt.savefig('ExampleDiagonal_3dsurface.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def classical_optimal_diag_surface(save=False):
    # Find matrices with a middle-ish band location and smaller band width
    poss_idx = np.where((band_loc < 0.7)*(band_loc > 0.3)*(h < 0.004) == True)[0]

    # This one is a matrix in the lower right corner of the winners plot where the model wins as well
    # poss_idx = np.where((band_loc < 0.2)*(h > 0.013) == True)[0]

    # Take the first one
    ex_idx = poss_idx[0]

    print(f'example_diag_surface: h = {h[ex_idx]}, band_loc = {band_loc[ex_idx]}')

    # Plot the diagonal
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # get the xy vector, this is a bit of a hack, we regenerate the matrix
    M = diag_A[ex_idx].shape[0]
    Nx = int(1 + np.sqrt(1 + M))
    Ny = int(M / Nx)
    _, xy, F = gsbm.getSmallBandMatrix(Nx, h[ex_idx], band_loc[ex_idx],False)

    X = xy[:,0].reshape(-1, Nx)
    Y = xy[:,1].reshape(-1, Nx)

    new_cmap = truncate_colormap(matplotlib.colormaps['Blues'], 0.3, 1.0)

    vmin = min(np.min(diag_opt_Dinv), np.min(diag_learn_Dinv))
    vmax = max(np.max(diag_opt_Dinv), np.max(diag_learn_Dinv))

    # surf = ax.plot_surface(X, Y, 2/(3*diag_A[ex_idx]).reshape(Ny,Nx), cmap=new_cmap)
    surf = ax.plot_surface(X, Y, diag_opt_Dinv[ex_idx].reshape(Ny,Nx), vmin=vmin, vmax=vmax, cmap=new_cmap)

    fig.set_figheight(10)
    fig.set_figwidth(12)
    fig.colorbar(surf, shrink=0.8)
    plt.title('$\omega_{co}$ Diagonal for h = 0.003, band location = 0.405')
    if save:
        plt.savefig('ExampleDiagonal_3dsurface_optimal.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def example_diag_2Dcolormap(save=False):
    # Find matrices with a middle-ish band location and smaller band width
    poss_idx = np.where((band_loc < 0.7)*(band_loc > 0.3)*(h < 0.004) == True)[0]

    # This one is a matrix in the lower right corner of the winners plot where the model wins as well
    # poss_idx = np.where((band_loc < 0.2)*(h > 0.013) == True)[0]

    # Take the first one
    ex_idx = poss_idx[0]

    # Plot the diagonal
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # get the xy vector, this is a bit of a hack, we regenerate the matrix
    M = diag_A[ex_idx].shape[0]
    Nx = int(1 + np.sqrt(1 + M))
    Ny = int(M / Nx)
    _, xy, _ = gsbm.getSmallBandMatrix(Nx, h[ex_idx], band_loc[ex_idx],False)

    X = xy[:,0].reshape(-1, Nx)
    Y = xy[:,1].reshape(-1, Nx)

    new_cmap = truncate_colormap(matplotlib.colormaps['Blues'], 0.3, 1.0)

    surf = ax.plot_surface(X, Y, diag_learn_Dinv[ex_idx].reshape(Ny,Nx), cmap=new_cmap)

    fig.set_figheight(10)
    fig.set_figwidth(12)
    fig.colorbar(surf)
    ax.set_proj_type('ortho')
    ax.elev = 90
    ax.azim = -90
    if save:
        plt.savefig('ExampleDiagonal_2dcolor.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
if __name__ == '__main__':
    save = True
    # save = False
    # eigenvalue_scatter_plots(save)
    # seperate_histograms(save)
    # stacked_histograms(save)
    # box_and_whisker(save)
    histograms_compared_to_learned(save)
    # winners_plot(save)
    # example_diag_horizontal(save)
    # example_diag_surface(save)
    # classical_optimal_diag_surface(save)
    # example_diag_2Dcolormap(save)