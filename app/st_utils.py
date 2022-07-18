import streamlit as st

import quantstats as qs
import FinanceDataReader as fdr

## custom libs
import conf, utils

ST_DATECACHE = {} # TODO: Make iterative datecache to support st momoization

class QS2ST:
    """Convert quantstats results to streamlit-displayable format
    """    
    def __init__(self, returns, benchmark=None):
        self.returns = returns
        self.benchmark = benchmark

        self.qs_plots = [f for f in dir(qs.plots) if f[0] != '_']
        self._plots = qs.reports._plots

        self.qs_stats = [f for f in dir(qs.stats) if f[0] != '_']
        self._stats = qs.stats
    
    def show_qsplot_info(self):
        return self.qs_plots
    
    def get_qsplot_fig(self,plot_name):
        assert plot_name in self.qs_plots

        fig_method = getattr(self._plots, plot_name)

        return fig_method(self.returns, show=False)
    
    def show_qsstats(self):
        return self.qs_stats

    def get_qsstats(self, stats_name):
        assert stats_name in self.qs_stats

        stats_method = getattr(self._stats, stats_name)

        return stats_method(self.returns)

@st.cache
def get_fdr_data(sid: str, start: int, end: int):
    if isinstance(start, int) or isinstance(end, int):
        start = utils.DateUtil.numdate2stddate(start)
        end = utils.DateUtil.numdate2stddate(end)
    
    assert start <= end
    assert utils.DateUtil.validate_date(start, min_date=conf.MIN_DATE)

    df = fdr.DataReader(sid, start, end)

    return df

    