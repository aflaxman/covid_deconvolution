import numpy as np, matplotlib.pyplot as plt, pandas as pd
import pymc as pm, scipy

def my_convolve(a, v):
    """Keep the length of `a`, but start with \sum_s a[0-s]*v[s]
    """
    b = np.convolve(a, v, mode='full')[:len(a)]
    return b


def my_weibull_kernel(amp, a, c):
    rv = scipy.stats.exponweib(a=a, c=c)

    # prevent underflow errors 
    x_min = min(21, int(np.ceil(rv.ppf(0.01))))
    x_max = min(21, int(np.floor(rv.ppf(0.99))))
    
    kernel_vals = amp * rv.pdf(range(x_min, x_max))
    return np.concatenate([np.zeros(x_min), kernel_vals, np.zeros(21-x_max)])


def apply_day_of_week_effect(s, beta, dow):
    """find day of week for s and multiply s[i] by beta[dow[i]]
    s : array-like to change
    beta : array-like to use for change
    dow : array-like indicating what day-of-week each entry of s is
    beta[0] = monday effect
    beta[6] = sunday effect
    """
    assert len(beta) == 7, 'expect effects for 7 days in the week'
    assert len(s) == len(dow), 'expect a dow for every entry of s'
    s_beta = beta[dow]
    t = s * np.exp(s_beta)
    return t


def simulate(s_infections_true, n_pop, symp_kernel_true, cc_kernel_true, cc_day_of_week_effects_true):
    df = pd.DataFrame(index=s_infections_true.index)
    
    df['n_pop'] = n_pop
    df['n_sampled'] = 1_000
    
    symp_intensity = my_convolve(s_infections_true, symp_kernel_true)
    df['n_symp'] = np.random.poisson(1e-3 + symp_intensity/df.n_pop * df.n_sampled)
    
    cc_intensity_smooth = my_convolve(s_infections_true, cc_kernel_true)
    dow = s_infections_true.index.map(pd.Timestamp.weekday).values
    cc_intensity = apply_day_of_week_effect(cc_intensity_smooth, cc_day_of_week_effects_true, dow)
    df['n_cc'] = np.random.poisson(cc_intensity)
                                     
    return df.iloc[21:]  # skip first three week


def model(data):
    X_smooth = [pm.Uniform(f'X_smooth_{i}', .1, data.n_pop,
                           value=np.clip(10*data.n_cc.iloc[i], 1, data.n_pop.iloc[i]))
                for i in range(len(data))]
    
    # potential to smooth X with
    # some appropriate smoothing prior
    @pm.potential
    def X_smooth_potential(X_smooth=X_smooth):
        mu = np.diff(np.log(X_smooth), n=1)
        return pm.normal_like(np.zeros_like(mu), mu, 500)
      
    # day of week effects 
    dow = data.index.map(pd.Timestamp.weekday).values
    beta_dow_infection = np.zeros(7)
    beta_dow_cc_nonzero = pm.Normal('beta_dow_cc_nonzero', 0, 100, value=np.zeros(4))
    @pm.deterministic(trace=True)
    def beta_dow_cc(beta_dow_cc_nonzero=beta_dow_cc_nonzero):
        return np.array([beta_dow_cc_nonzero[0], 0, 0, 0,
                         beta_dow_cc_nonzero[1], beta_dow_cc_nonzero[2], beta_dow_cc_nonzero[3]])

    # latent state: number of new infections each day
    @pm.deterministic(trace=True)
    def X(X_smooth=X_smooth, beta_dow_infection=beta_dow_infection):
        return apply_day_of_week_effect(
            X_smooth, beta_dow_infection, dow)
        
    # symptom kernel parameterized by Weibull distribution
    symp_kernel_amp = pm.Uniform('symp_kernel_amp', .1, 10, value=5)
    symp_kernel_knots = [0] + [pm.Uniform(f'symp_kernel_knot_{i}', 0, .1, value=.01) for i in range(6)]
    @pm.deterministic(trace=True)
    def symp_kernel(symp_kernel_amp=symp_kernel_amp, symp_kernel_knots=symp_kernel_knots):
        ii = np.linspace(0,7,num=21, endpoint=False)
        ii = np.array(np.floor(ii), dtype=int)
        kernel_shape = np.array(symp_kernel_knots)[ii]
        return symp_kernel_amp * kernel_shape / kernel_shape.sum()

    @pm.potential
    def symp_days_to_infection_ratio(symp_kernel=symp_kernel):
        return pm.normal_like(symp_kernel.sum(), 5, .2**-2)  # TODO: evaluate evidence base for this prior, refactor into symp_kernel_amp prior

    @pm.deterministic(trace=True)
    def mu_symp(X=X, symp_kernel=symp_kernel):
        # add small offset to prevent zero-probability error
        return 1e-3 + my_convolve(X, symp_kernel)/data.n_pop * data.n_sampled

    @pm.observed
    def y_symp(mu=mu_symp, value=data.n_symp):
        return pm.poisson_like(value[10:], mu[10:])
    
    cc_kernel_amp = pm.Uniform('cc_kernel_amp', .01, .99, value=.1)
    cc_kernel_knots = [0, 0] + [pm.Uniform(f'cc_kernel_knot_{i}', 0, .1, value=.01) for i in range(5)]
    @pm.deterministic(trace=True)
    def cc_kernel(cc_kernel_amp=cc_kernel_amp, cc_kernel_knots=cc_kernel_knots):
        ii = np.linspace(0,7,num=21, endpoint=False)
        ii = np.array(np.floor(ii), dtype=int)
        kernel_shape = np.array(cc_kernel_knots)[ii]
        return cc_kernel_amp * kernel_shape / kernel_shape.sum()

    @pm.potential
    def cc_to_infection_ratio(cc_kernel=cc_kernel):
        return pm.normal_like(cc_kernel.sum(), 0.1, .02**-2)  # TODO: evaluate evidence base for this prior, refactor into cc_kernel_amp prior
    
    @pm.deterministic(trace=True)
    def mu_cc_smooth(X=X, cc_kernel=cc_kernel):
        return my_convolve(X, cc_kernel)
    @pm.deterministic(trace=True)
    def mu_cc(mu_cc_smooth=mu_cc_smooth, beta_dow_cc=beta_dow_cc):
        return apply_day_of_week_effect(
            mu_cc_smooth, beta_dow_cc, dow) + 1e-3 # add small offset to prevent zero-probability error
    @pm.observed
    def y_cc(mu=mu_cc, value=np.clip(data.n_cc, 0, data.n_pop)):
        return pm.poisson_like(value[10:], mu[10:])
    
    return locals()

def fit(var_dict, verbose=False, fast=False):
    m = pm.MCMC(var_dict)
    m.use_step_method(pm.AdaptiveMetropolis, m.symp_kernel_knots[1:])
    m.use_step_method(pm.AdaptiveMetropolis, m.cc_kernel_knots[2:])
    for i in range(0, len(m.X_smooth)-7, 3):
        m.use_step_method(pm.AdaptiveMetropolis, m.X_smooth[i:(i+7)])

    for i in range(5):
        pm.MAP([m.y_cc, m.y_symp, m.X_smooth,
                m.X_smooth_potential]).fit(method='fmin_powell', verbose=verbose)
        pm.MAP([m.y_cc, m.y_symp, m.cc_kernel_knots, m.cc_kernel_amp, m.cc_to_infection_ratio,
               ]).fit(method='fmin_powell', verbose=verbose)
        pm.MAP([m.y_cc, m.y_symp, m.symp_kernel_knots, m.symp_kernel_amp, m.symp_days_to_infection_ratio,
               ]).fit(method='fmin_powell', verbose=verbose)
        pm.MAP([m.y_cc, m.y_symp, m.X_smooth,
                m.X_smooth_potential, m.beta_dow_cc_nonzero]).fit(method='fmin_powell', verbose=1)

    pm.MAP(var_dict).fit(method='fmin_powell', verbose=1)

    if fast:
        m.sample(10)
    else:
        m.sample(50_000, 20_000, 30)

    return m


def my_plot(stoch, xx, color='C0', label='Predicted'):
    s_mean = stoch.trace().mean(axis=0)
    s_lb, s_ub = pm.utils.hpd(stoch.trace(), 0.05)
    plt.plot(xx, s_mean, '-', color=color, label=label, zorder=3)
    plt.plot(xx, s_lb, '--', color=color, zorder=3)
    plt.plot(xx, s_ub, '--', color=color, zorder=3)


def etl_state(loc_name, loc_abbr, df_full_data, df_fb_data, df_zip):
    """Create input data for selected state

    To load data files, try::

        df_full_data = pd.read_csv('/ihme/covid-19/model-inputs/best/full_data.csv')
        df_fb_data = pd.read_csv('/home/j/Project/simulation_science/covid/data/fb_data_usa.csv', index_col=0, low_memory=False)
        df_zip = pd.read_csv('/home/j/Project/simulation_science/covid/data/ZIP-COUNTY-FIPS_2018-03.csv')
    """
    data = pd.DataFrame()

    # first two columns come from IHME COVID-19 Projection Model Inputs
    df = df_full_data
    t = df[df['Province/State'] == loc_name]
    data['n_pop'] = t['population']
    data['n_cc'] = t['Confirmed'].diff()
    data.index = t['Date'].map(pd.Timestamp)

    # next two columns come from FB data
    df = df_fb_data
    if not 'zip5' in df.columns:
        def to_five(s):
            if pd.isna(s):
                return np.nan
            return int(str(s)[:5])
        df['zip5'] = df.zipcode.map(to_five)
    if not 'state' in df.columns:
        df['state'] = df.zip5.fillna(0).map(df_zip.groupby('ZIP').STATE.first())

    # TODO: search systematically for informative syndromes
    df['no_smell_not_congested'] = df.smell_taste_loss & ~df.nasal_congest

    t = df.groupby(['state', 'date']).no_smell_not_congested.describe().filter(['count', 'mean'])
    t = t.reset_index()
    t = t[t.state == loc_abbr]
    t.index = t.date.map(pd.Timestamp)

    data['n_sampled'] = t['count']
    data['n_symp'] = t['count']*t['mean']
    data = data.dropna()

    return data


def summarize_results(m):
    out_data = m.data.filter(['n_cc', 'n_symp']).copy()
    out_data['cases_wo_day_of_week_effect'] = m.mu_cc_smooth.trace().mean(axis=0)
    out_data['smoothed_symptomatic_infections'] = m.mu_symp.trace().mean(axis=0)
    out_data['new_infections'] = m.X.trace().mean(axis=0)
    out_data = out_data.iloc[14:]
    return out_data


def make_plots(m):
    plt.figure(figsize=(8.25, 4.), dpi=120)
    my_plot(m.X, m.data.index)
    plt.grid()
    plt.ylabel('New Infections')
    plt.axis(ymin=0)


    plt.figure(figsize=(8.25, 4.), dpi=120)
    my_plot(m.mu_symp, m.data.index)
    m.data.n_symp.plot(label='Observed', color='grey', marker='s', zorder=2)
    
    plt.legend(loc=(1.01, .01))
    plt.grid()
    plt.ylabel('Symptomatic in Sample')


    plt.figure(figsize=(8.25, 4.), dpi=120)

    my_plot(m.mu_cc, m.data.index, 'C0', 'Predicted')
    my_plot(m.mu_cc_smooth, m.data.index, 'C1', 'Without day-of-week effect')
    m.data.n_cc.plot(label='Observed', color='grey', marker='s', zorder=2)
    
    plt.legend(loc=(1.01, .01))
    plt.grid()
    plt.ylabel('New Confirmed Cases')


    plt.figure(figsize=(8.25, 4.), dpi=120)

    my_plot(m.beta_dow_cc, range(7))

    plt.grid()
    plt.ylabel('Day of week effects')


    plt.figure(figsize=(8.25, 4.), dpi=120)
    my_plot(m.cc_kernel, range(21), 'C0', 'Convolution kernel for confirmed cases')
    plt.grid()
    my_plot(m.symp_kernel, range(21), 'C1', 'Convolution kernel for symptoms')

    plt.legend(loc=(1.01, .01))

