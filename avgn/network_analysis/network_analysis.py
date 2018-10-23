import numpy as np
from tqdm import tqdm_notebook as tqdm
import time
import pandas as pd
from datetime import datetime, timedelta
from datetime import time as dt_time


from tempfile import NamedTemporaryFile

import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
import matplotlib.dates as dates


def generate_manifold(model,dims, iter_, num_examples,batch_size,num_gpus):
    """ get output for entire manifold
    """
    # allocate collector
    next_batch = iter_.__next__()
    total_batch = int(np.ceil(num_examples/batch_size/num_gpus))
    output_data = [model.sess.run((model.z_x), {model.x_input: next_batch[0]})]
    all_data =  [np.zeros([i if ii !=0 else total_batch*batch_size for ii,i in enumerate(np.shape(i))], dtype=np.float32) for i in  output_data]

    # fill collector
    for i in tqdm(range(total_batch), leave=False):
        next_batch = iter_.__next__()
        output_data = [model.sess.run((model.z_x), {model.x_input: next_batch[0]})]
        for item in range(len(output_data)):
            all_data[i*batch_size:(i+1)*batch_size] = (output_data[item]).astype(np.float32)
    return all_data[:num_examples]

def z_score(X):
    return (X - np.mean(X))/np.std(X)

def inv_z_score(Y, X):
    return (Y * np.std(X)) + np.mean(X)

def make_grid(x_lims, y_lims , maxx = 28*4, maxy = 28*4):
    """ Makes a grid to perform volumetric analysis over
    """
    pts = []
    xs, hx = np.linspace(x_lims[0], x_lims[1], maxx, retstep=True)
    ys, hy = np.linspace(y_lims[0], y_lims[1], maxy, retstep=True)
    for x in xs:
        for y in ys:
            pts.append([x, y])
    return maxx, maxy, hx,hy, pts


def metric_and_volume(model, maxx, maxy, hx,hy, pts, dims, batch_size):
    # decode points from grid
    PTS = np.empty((len(pts), dims[0]*dims[1]))
    for batchnum in tqdm(range(int(maxx*maxy / batch_size)), leave=False):
        batchpts = pts[batchnum*batch_size:batchnum*batch_size+batch_size]
        batchPTS = model.sess.run((model.x_tilde), {model.z_x: batchpts})
        PTS[batchnum*batch_size:batchnum*batch_size+batch_size] = batchPTS

    mats = np.zeros((maxx, maxy, 2, 2))
    dets = np.zeros((maxx, maxy))
    dmet = np.zeros((maxx, maxy, 2, 2, 2))

    for x_i, y_i in tqdm([(x_i, y_i) for x_i in range(1, maxx-1) for y_i in range(1, maxy-1)], leave=False):
        i1 = maxy*(x_i - 1)+y_i
        i2 = maxy*(x_i + 1)+y_i
        i3 = maxy*x_i + (y_i-1)
        i4 = maxy*x_i + (y_i+1)

        i5 = maxy*(x_i - 1) + (y_i-1) # x-1, y-1
        i6 = maxy*(x_i - 1) + (y_i+1)  # x-1, y+1
        i7 = maxy*(x_i + 1) + (y_i-1) # x+1, y-1
        i8 = maxy*(x_i + 1) + (y_i+1) # x+1, y+1

        i0 = maxy*x_i + y_i
        mx = PTS[i1, :]
        px = PTS[i2, :]
        my = PTS[i3, :]
        py = PTS[i4, :]

        # first partials
        dx = (px - mx)/(2*hx)
        dy = (py - my)/(2*hy)

        # second partials
        rxx = (px + mx - 2*PTS[i0, :])/(hx**2)
        ryy = (py +my - 2*PTS[i0, :])/(hy**2)
        rxy = (PTS[i8, :] - PTS[i7, :] - PTS[i6, :] + PTS[i5, :])/(4*hx*hy)

        # Normal vector
        d1a = np.dot(dx, dx)
        d1b = np.dot(dy, dy)
        d2a = np.dot(dx, dy)
        d2b = np.dot(dx, dy)

        mats[x_i, y_i, 0, 0] = np.dot(dx, dx)
        mats[x_i, y_i, 1, 1] = np.dot(dy, dy)
        mats[x_i, y_i, 0, 1] = np.dot(dx, dy)
        mats[x_i, y_i, 1, 0] = np.dot(dx, dy)


        #first derivatives of metric
        dmet[x_i, y_i, 0, :, :] = (mats[x_i+1, y_i, :, :] - mats[x_i-1, y_i, :, :])/(2*hx)
        dmet[x_i, y_i, 1, :, :] = (mats[x_i, y_i+1, :, :] - mats[x_i, y_i-1, :, :])/(2*hy)

        dets[x_i, y_i] = d1a*d1b-d2a*d2b

    #Christoffels
    #christoffel = (1/2)*(np.einsum('abkij->abjki', dmet) + np.einsum('abkij->abikj', dmet) - np.einsum('abkij->abkij', dmet))
    return dets

# This is code for turning sequences of images into a GIF, and displaying it in jupyter
IMG_TAG = """<img src="data:image/gif;base64,{0}" alt="some_text">"""

def anim_to_gif(anim, fps):
    data="0"
    with NamedTemporaryFile(suffix='.gif') as f:
        anim.save(f.name, writer='imagemagick', fps=fps);
        data = open(f.name, "rb").read()
        data = data.encode("base64")
    return IMG_TAG.format(data)

def display_animation(anim, fps=20):
    plt.close(anim._fig)
    return HTML(anim_to_gif(anim, fps=fps))


def cluster_data(data, algorithm, args, kwds, verbose = True):
    """ Cluster data using arbitrary clustering algoritm in sklearn
    """
    # Function modified from HDBSCAN python package website
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    if verbose: print('Clustering took {:.2f} s'.format(end_time - start_time))
    return labels



def split_seq_by_time(times, idxs, max_timedelta = 30):
    """ splits up time indexes
    """
    idxs_sorted = idxs[times.argsort()]
    times.sort()
    time_before = np.concatenate(
        ([0.],[(times[i] - times[i-1])/np.timedelta64(1, 's')
          for i in np.arange(1,len(times))]))
    sequence_breaks = np.unique(np.concatenate((
                np.where(time_before > max_timedelta)[0], np.array([0,len(times)]))))
    idx_seqs = [idxs_sorted[sequence_breaks[i]:sequence_breaks[i+1]] for
           i in range(len(sequence_breaks[:-1]))]
    return idx_seqs



def split_times_into_seqs(BirdData, max_timedelta, seq_len_cutoff):
    """
    """
    sequence_num = np.zeros(len(BirdData))-2
    day_num = np.zeros(len(BirdData))-2
    sequence_syllable = np.zeros(len(BirdData))-2

    short_seqs =  np.zeros(len(BirdData))-2
    BirdData.sort_values(by='syllable_time')
    BirdData.index = np.arange(len(BirdData))

    seq_num_tot = 0
    #For each bird label the day
    bird_dates = [i.date() for i in BirdData['syllable_time']]
    for i,date in enumerate(tqdm(np.unique(bird_dates), leave=False)):
        day_num[np.array(date) == bird_dates] = i
    BirdData['day_num'] = day_num

    # For each bird label the sequence number
    bird_times = BirdData['syllable_time']
    idx_seqs = split_seq_by_time(np.array(bird_times.values),
                                 np.array(bird_times.index),
                                 max_timedelta=max_timedelta)

    for seq_i, idxs in enumerate(tqdm(idx_seqs, leave=False)):
        sequence_num[idxs] = seq_num_tot
        seq_num_tot+=1
        # rm short seqs
        if len(idxs) < seq_len_cutoff:
            short_seqs[idxs] = 1

        sequence_syllable[idxs]= np.arange(len(idxs)) # Label the syllable number

    BirdData['sequence_syllable'] =sequence_syllable
    BirdData['sequence_num'] = sequence_num

    BirdData = BirdData[short_seqs != 1]

    return BirdData

def syllables_to_sequences(bird_data, to_analyze):
    """ Take a list of bird data, saves a pickle of a dataframe of those sequences
    """
    all_sequences = pd.DataFrame(columns=['sequence_num', 'sequence_type'] + to_analyze)
    for cur_seq in tqdm(np.unique(bird_data['sequence_num']), leave=False):
        sequence_data = bird_data[bird_data['sequence_num'] == cur_seq].sort_values('sequence_syllable')
        all_sequences.loc[len(all_sequences)] = [cur_seq, to_analyze] +[sequence_data[i].values for i in to_analyze]
    return all_sequences


##################### Plot signing behavior #####################


def create_hourly_song_heatmap(BirdData):
    """ Plot a heatmap of when the bird is singing
    """
    # Create the heatmap
    BirdData.index = BirdData.syllable_time
    BirdData['hour'] = BirdData.syllable_time.map(lambda x: x.hour)
    freq = 'D' # could also be 'W' (week) or 'D' (day), but month looks nice.
    BirdData = BirdData.set_index('syllable_time', drop=False)

    BirdData.index = BirdData.index.to_period(freq)
    mindate = BirdData.index.min()
    maxdate = BirdData.index.max()

    pr = pd.period_range(mindate, maxdate, freq=freq)
    hm = pd.DataFrame(np.zeros([len(pr), 24]) , index=pr)
    for period in pr:
        # HERE'S where the magic happens...with pandas, when you structure your data correctly,
        # it can be so terse that you almost aren't sure the program does what it says it does...
        # For this period (month), find relevant emails and count how many emails were received in
        # each hour of the day. Takes more words to explain than to code.
        if period in BirdData.index:
            hm.ix[period] = BirdData.ix[[period]].hour.value_counts()
    #hm.fillna(0, inplace=True)
    total_email = BirdData.groupby(level=0).hour.count()
    return hm,pr, total_email

from matplotlib.colors import ColorConverter, ListedColormap
import matplotlib.colors as colors

def plot_song_heatmap(hm,pr, total_email):
    ### Set up figure
    fig = plt.figure(figsize=(12,8))
    # This will be useful laterz
    gs = gridspec.GridSpec(2, 2, height_ratios=[4,1], width_ratios=[20,1],)
    gs.update(wspace=0.05)

    ### Plot our heatmap
    ax = plt.subplot(gs[0])
    x = dates.date2num([p.start_time for p in pr])
    t = [datetime(2000, 1, 1, h, 0, 0) for h in range(24)]
    t.append(datetime(2000, 1, 2, 0, 0, 0)) # add last fencepost
    y = dates.date2num(t)
    cm = plt.get_cmap('Oranges')
    cm_uni  = colors.ListedColormap(['#EFEFEF'])

    plt.pcolormesh(x, y, np.zeros(np.shape(hm.transpose().as_matrix())), cmap=cm_uni, edgecolors='white', linewidth=3)
    #plt.pcolor(hm.transpose()*0, vmin=0, vmax=1, cmap=ListedColormap(['#EEEEEE']))

    hm[hm==0] = np.inf
    hm = np.ma.masked_invalid(np.atleast_2d(hm))
    fill_data = np.ma.masked_where(np.isnan(hm), hm)
    plt.pcolor(x, y, hm.transpose(), cmap=cm, edgecolors='white', linewidth=3)



    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ### Now format our axes to be human-readable
    ax.xaxis.set_major_formatter(dates.DateFormatter('%d/%m/%y'))
    ax.yaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    ax.set_yticks(t[::2])
    ax.set_xticks(x[::7])
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([t[0], t[-1]])
    ax.tick_params(axis='x', pad=14, length=10, direction='inout')

    ### pcolor makes it sooo easy to add a color bar!
    #plt.colorbar(cax=plt.subplot(gs[1]))

    ax2 = plt.subplot(gs[2])
    #plt.plot_date(total_email.index, total_email, '-', linewidth=1.5, color=cm(0.999))
    #ax2.fill_between(total_email.index, 0, total_email, color=cm(0.5))

    ax2.xaxis.tick_top()
    x = range(len(total_email.values))
    ax2.plot(x, total_email.values,color=cm(0.999))
    ax2.fill_between(x, 0, total_email, color=cm(0.5))
    ax2.set_xlim((0, len(x)-1))
    ax2.tick_params(axis='x', pad=14, length=10, direction='inout')
    #out = ax2.set_xticks(total_email.index[::12])
    out = ax2.xaxis.set_ticklabels([])

def generate_specs_from_z_path(model, z_path, dims, batch_size):
    num_batches = int(np.ceil(float(len(z_path))/batch_size))
    all_x = np.zeros((len(z_path), dims[0]*dims[1]))
    for batch in range(num_batches):
        batch_input = np.zeros((batch_size, 2))
        batch_to_enter = z_path[batch*batch_size:(batch+1)*batch_size]
        batch_input[:len(batch_to_enter), :] = batch_to_enter
        all_x[batch*batch_size:(batch*batch_size)+len(batch_to_enter)] = model.sess.run((model.x_tilde), {model.z_x: batch_input})[:len(batch_to_enter)]
    return all_x


def draw_grid(model,dims,batch_size, xs, ys,spacing = .25, zoom = 1, savefig=False):
    """ Draw a grid from spectrograms
    """
    x = np.linspace(xs[0],xs[1], np.abs(xs[1]-xs[0])/spacing)
    y = np.linspace(ys[0],ys[1], np.abs(ys[1]-ys[0])/spacing)
    xv, yv = np.meshgrid(x, y)
    #
    spectrograms_list = generate_specs_from_z_path(model,np.stack((np.ndarray.flatten(xv), np.ndarray.flatten(yv))).T, dims, batch_size)
    spectrograms_list = np.reshape(spectrograms_list,(len(spectrograms_list), dims[0], dims[1]))
    spectrograms_list = np.reshape(spectrograms_list,tuple(list(np.shape(xv)) + [dims[0]] + [dims[1]]))
    x_len = np.shape(spectrograms_list)[0]
    y_len = np.shape(spectrograms_list)[1]
    canvas = np.zeros((x_len*dims[0], y_len*dims[1]))
    fig_spc, ax_spc = plt.subplots(figsize=(y_len*zoom,x_len*zoom))
    ax_spc.axis('off')


    print(np.shape(canvas))
    zoom = 1
    for i in range(y_len):
        for j in range(x_len):
            canvas[j*dims[1]:(j+1)*dims[1],i*dims[0]:(i+1)*dims[0]] =np.flipud(spectrograms_list[j,i,:,:])
            if (i ==0) : ax_spc.axhline(j*dims[0]-1, color='k', lw=2)
        ax_spc.axvline(i*dims[0]-1, color='k', lw=2)
    #ax_spc.axhline((j+1)*dims[0]-2, color='k', lw=4)
    #ax_spc.axvline((i+1)*dims[0]-2, color='k', lw=4)
    [j.set_linewidth(2) for j in ax_spc.spines.values()]
    [j.set_edgecolor('k') for j in ax_spc.spines.values()]

    ax_spc.matshow(canvas, cmap=plt.cm.Greys, interpolation='nearest', aspect='auto')
    plt.setp( ax_spc.get_xticklabels(), visible=False)
    plt.setp( ax_spc.get_yticklabels(), visible=False)
    plt.show()
    # save figure
    if savefig:
        if not os.path.exists('../../data/imgs/interpolation/'+species+'/'):
            os.makedirs('../../data/imgs/interpolation/'+species+'/')
            #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_spc.savefig('../../data/imgs/interpolation/'+species+'/'+bird+'_indv.png', bbox_inches='tight', transparent=True)
    return xv, yv


####

def colorline(
    x, y,ax, z=None,cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def draw_transition_diagram(bird_seqs, cmap= plt.get_cmap('cubehelix'), alpha=0.05 , num_ex = 30, linewidth=3, ax = None):
    if ax==None:
        fig, ax= plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    for i in tqdm(range(num_ex)):
        seq_dat = np.reshape(np.concatenate((bird_seqs[i])), (len(bird_seqs[i]),2))
        colorline(seq_dat[:,0], seq_dat[:,1],ax, np.linspace(0, 1, len(seq_dat[:,1])), cmap=cmap, linewidth=linewidth, alpha = alpha)

    z = np.concatenate([np.vstack(i) for i in bird_seqs])
    minx = np.min(z[:,0]); maxx = np.max(z[:,0])
    miny = np.min(z[:,1]); maxy = np.max(z[:,1])
    ax.set_xlim((minx,maxx))
    ax.set_ylim((miny,maxy))
    ax.axis('off')
