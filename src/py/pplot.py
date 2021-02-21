 #!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import fnmatch
import argparse
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# values
col_dict = {
    "red" : np.array([251, 180, 174])/255.0,
    "blue" : np.array([179, 205, 227])/255.0,
    "green" : np.array([204, 235, 197])/255.0,
    "purple" : np.array([222, 203, 228])/255.0,
    "orange" : np.array([254, 217, 166])/255.0,
    "yellow" : np.array([255, 255, 204])/255.0,
    "brown" : np.array([229, 216, 189])/255.0,
    "pink" : np.array([253, 218, 236])/255.0,
    "white" : np.array([255, 255, 255])/255.0,
    "pgray" : np.array([240, 240, 240])/255.0,
    "gray" : np.array([150, 150, 150])/255.0
}
# colors pick order
colors = ["blue", "red", "green", "purple", "orange",
          "yellow", "brown", "ping", "white", "gray", "black"]
# markers order
markers = ['^', 'o', 's', '*', '+']

parser = argparse.ArgumentParser("pplot")
parser.add_argument('-i', '--rid', default=[], action='append', help='.npy files')
parser.add_argument('-n', '--name', default=[], action='append', help='conf names')
parser.add_argument('-l', '--ls', action='store_true', default=False, help='list metrics in dir')
parser.add_argument('-s', '--show', action='store_true', default=False, help='show metrics')
parser.add_argument('-m', '--maxit', default=-1, type=int, help='max num iterations')
args = parser.parse_args()

base_path, _ = os.path.split(os.path.realpath(__file__))
base_path = base_path + "/../../dataset/metrics/"

if args.ls:
    mets_list = os.listdir(base_path)
    print("\n-- listing:")
    for i, m in enumerate(mets_list):
        if os.path.isdir(os.path.join(base_path, m)):
            print(' -', m)
    exit()

num_confs = len(args.rid)
bar_width = 1.0/num_confs
conf_names = [r'Cfg. ' + str(i) for i in range(num_confs)] if len(args.name) == 0 else args.name
assert len(conf_names) == num_confs, "input dimensions mismatch"

save_paths = [os.path.join(base_path, m + '/metrics/') for m in args.rid]
ae_metrics = [np.load(base_path + m + "/ae_mets.npy") for m in args.rid
              if os.path.isfile(base_path + m + "/ae_mets.npy")]
encodings_accuracy = [np.load(base_path + m + "/encoding_accuracy.npy") for m in args.rid
                      if os.path.isfile(base_path + m + "/encoding_accuracy.npy")]
# gan_metrics = [np.load(base_path + m + "/gan-tf_mets.npy") for m in args.rid
#                if os.path.isfile(base_path + m + "/gan-tf_mets.npy")]
gan_metrics = [np.load(base_path + m + "/generation-loss.npy") for m in args.rid
               if os.path.isfile(base_path + m + "/generation-loss.npy")]
generation_accuracy = [np.load(base_path + m + "/generation_accuracy.npy") for m in args.rid
                      if os.path.isfile(base_path + m + "/generation_accuracy.npy")]
tf_accuracy = [np.load(base_path + m + "/tf_accuracy.npy") for m in args.rid
               if os.path.isfile(base_path + m + "/tf_accuracy.npy")]
update_time = [np.load(base_path + m + "/update_time.npy")[:, 0] for m in args.rid
               if os.path.isfile(base_path + m + "/update_time.npy")]

iter_num = min([x.shape[0] for x in update_time])
iter_num = min(iter_num, args.maxit) if args.maxit != -1 else iter_num


def barchart(idx, value, blabel="", bcolor=col_dict["blue"], alpha_ch=0.95):
    bscale = 0.5
    xts = np.arange(1) - 0.5*num_confs*bscale*bar_width + bscale*bar_width*idx
    plt.bar(xts, value[0], 0.9*bscale*bar_width, yerr=value[1],
            color=bcolor, alpha=alpha_ch, align='center',
            edgecolor='black', linewidth=0.07, label=blabel,
            error_kw=dict(ecolor=0.8*bcolor, lw=1, capsize=5, capthick=2))


def pplot(value, blabel="", bcolor=col_dict["blue"], col_shade=0.9, line_width=1.2,
          bmarker=markers[0], marker_sz=7, mark_at=50, compute_moving_mean=True):
    if compute_moving_mean:
        plt.plot(value, lw=0.8, linestyle='-', color=0.99*bcolor)

        moving_window_sz = 50
        moving_mean = np.array([np.mean(value[slice(None if m - moving_window_sz < 0 else m - moving_window_sz, m, None)])
                                for m in range(0, value.shape[0], 1)])

        plt.plot(moving_mean, lw=line_width, linestyle='-', color=col_shade*bcolor,
                 marker=bmarker, markersize=marker_sz, markevery=mark_at, label=blabel)
    else:
        plt.plot(value, lw=line_width, linestyle='-', color=col_shade*bcolor,
                 marker=bmarker, markersize=marker_sz, markevery=mark_at, label=blabel)




def configure_plot(p, xticks=None, title="title", y_label="label", x_label="\#iter",
                   font_sz=18, xt_step=1, leg_loc='lower right', save_dir='', out_name=''):
    # font_sz = max(12, font_sz)
    # p.legend(loc=leg_loc, fontsize=font_sz - 4, ncol=3)
    p.xlabel(x_label, fontsize=font_sz)
    # p.ylabel(r'' + y_label, fontsize=font_sz - 4)
    p.yticks(fontsize=font_sz - 4)
    p.title(r'' + title, fontsize=font_sz)
    p.tight_layout()
    p.grid(color=np.array([210, 210, 210])/255.0, linestyle='--', linewidth=1)

    ax = p.gca()
    ax.grid(color='gainsboro', linestyle='--', linewidth=0.1, alpha=0.5)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis='y', useOffset=False)

    if x_label == "":
        ax.set_xticks([], [])

    if xticks is not None:
        ax.set_xticks(xticks[::xt_step] - 0.25*bar_width)
        ax.set_xticklabels([r'' + " " + str(i) for i in xticks[::xt_step]], fontsize=font_sz - 2)

    if len(save_dir) != 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(out_name) == 0:
            out_name = title + "_" + y_label + ".pdf"
            out_name = out_name.lower().replace("]", "").replace("[", "").replace(" ", "_")
        p.savefig(os.path.join(save_dir, out_name), format='pdf')

############################## Update Time profiles
# if len(update_time) != 0:
#     plt.figure()
#     for i in range(num_confs):
#         # [1:] to remove the first update (initialization update)
#         cfg_val = [np.mean(update_time[i][1:iter_num]), np.std(update_time[i][1:iter_num])]
#         barchart(i, cfg_val, blabel=conf_names[i], bcolor=col_dict[colors[i]])
#     configure_plot(plt, title="Update time", y_label="[sec]", x_label="", leg_loc='upper right')

############################## AutoEncoder
if len(ae_metrics) != 0:
    plt.figure(figsize=(7, 8))
    plt.subplot(2, 1, 1)
    for i in range(num_confs):
        pplot(ae_metrics[i][:iter_num, 0], blabel=conf_names[i], bcolor=col_dict[colors[i]], compute_moving_mean=False)
    cfg_val = [np.mean(update_time[0][1:iter_num]), np.std(update_time[0][1:iter_num])]
    configure_plot(plt, title="Loss", leg_loc='upper left') #, save_dir=save_paths[i])

if len(encodings_accuracy) != 0:
    plt.subplot(2, 1, 2)
    color = 'orange'
    for i in range(num_confs):
        pplot(encodings_accuracy[i][:iter_num, 0], blabel=conf_names[i], bcolor=col_dict[color])
    configure_plot(plt, title="Mean Abs. Error [m]", leg_loc='upper left', save_dir=save_paths[i], out_name='encoding_metrics.pdf')

############################## TF projector
# if len(gan_metrics) != 0:
#     plt.figure()
#     for i in range(num_confs):
#         pplot(gan_metrics[i][:iter_num, 0], blabel=conf_names[i], bcolor=col_dict[colors[i]])
#     configure_plot(plt, title="Transform projector", y_label="Loss", leg_loc='upper left', save_dir=save_paths[i])

# if len(tf_accuracy) != 0:
#     plt.figure()
#     for i in range(num_confs):
#         pplot(tf_accuracy[i][:iter_num, 0], blabel=conf_names[i], bcolor=col_dict[colors[i]])
#     configure_plot(plt, title="Transform projector", y_label="Error", leg_loc='upper left', save_dir=save_paths[i])

############################## GAN
if len(gan_metrics) != 0:
    plt.figure(figsize=(7, 8))
    plt.subplot(2, 1, 1)
    # import pdb; pdb.set_trace()

    for i in range(num_confs):
        pplot(gan_metrics[i][:iter_num-10],
              blabel=conf_names[i] + " Dis", bcolor=col_dict['blue'], bmarker=markers[0])
        pplot(gan_metrics[i][500:500+iter_num],
              blabel=conf_names[i] + " Gen", bcolor=col_dict['red'], bmarker=markers[3])
        configure_plot(plt, title="Loss", leg_loc='upper left') #, save_dir=save_paths[i])

if len(generation_accuracy) != 0:
    plt.subplot(2, 1, 2)
    color = 'orange'
    for i in range(num_confs):
        pplot(generation_accuracy[i][:iter_num, 0],
              blabel=conf_names[i] + " Gen", bcolor=col_dict[color], bmarker=markers[3])
    configure_plot(plt, title="Mean Abs. Error [m]",
                   leg_loc='upper left', save_dir=save_paths[i], out_name='generation_metrics.pdf')

if args.show:
    plt.show()
