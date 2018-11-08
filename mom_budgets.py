# Some tools to help with budget calculations in mom5 (I tried to keep the naming consistent with the MOM5 manual [link?])
import xarray as xr
from xgcm import Grid
import numpy as np
import os
from xarrayutils.cm26_codebucket import drop_all_coords

################# more high level convenience functions

def split_adv_budget(ds):
    ds = ds.copy()
    grid=Grid(ds)
    area = ds.area_t
    div_x = - grid.diff(ds.o2_xflux_adv, 'X', boundary='fill') / area
    div_y = - grid.diff(ds.o2_yflux_adv, 'Y', boundary='fill') / area
    div_z =  grid.diff(ds.o2_zflux_adv, 'Z', boundary='fill') / area

    for data, name in zip([div_x, div_y, div_z], ['o2_advection_%s' % a for a in ['x', 'y', 'z']]):
        ds[name] = data
    return ds

def budget_prep(ds, tracer='o2'):
    """Does some simplifications of the budget to compare high res and low res better
    This works for o2 but namingconventions might be different for other tracers...use with care"""
    ds = ds.copy()
    # combine vertical terms
    ds['%s_diff_ver' %tracer] = ds['%s_nonlocal_KPP' %tracer] + ds['%s_vdiffuse_impl' %tracer]

    # combine fields from the two resolutions into comparable fields
    if 'neutral_diffusion_%s' %tracer in ds.data_vars:
        ds['%s_diff_hor' %tracer] = ds['neutral_diffusion_%s' %tracer] + ds['neutral_gm_%s' %tracer]
        ds['%s_diff' %tracer] = ds['%s_diff_ver' %tracer] + ds['%s_diff_hor' %tracer] + ds['%s_rivermix' %tracer]
    else:
        ds['%s_diff' %tracer] = ds['%s_diff_ver' %tracer]

    ds['%s_residual' %tracer] = ds['%s_tendency' %tracer] - (ds['%s_advection' %tracer] + ds['j%s' %tracer] + ds['%s_diff' %tracer] + ds['%s_submeso' %tracer])
    return ds

################### Bare bones budget calculations

def calculate_dzstar(ds, dim='st'):
    """Creates static thickness (dzstar) for MOM5 tracer cells.
    Note that the cell thickness for u cells is defined as min of the sourrounding cells."""
    z = ds['%s_ocean' %dim]
    diff_dim = '%s_edges_ocean' %dim
    dz_raw = ds[diff_dim].diff(diff_dim).data
    return xr.DataArray(dz_raw, coords=z.coords)

def calculate_dz(eta, h, dzstar):
    """reconstructs cell thickness dz(x,y,z,t) from sea surface elevation ('eta') and the full ocean depth ('h')
    and fixed thickness levels (dzstar)."""
    dz = dzstar*(1+(eta/h))
    return dz

# Legacy version (deprecate soon)
def reconstruct_thickness(eta, h, mask, dz_star, ref_array):
    """reconstructs cell thickness dz(x,y,z,t) from sea surface elevation ('eta') and the full ocean depth ('h')
    and fixed thickness levels (dz_star). ref has to be a full (x,y,z,t) array to carry 3d masking"""
    print('Old version with masking implemented. Change to `calculate_dzt` and mask afterwards')
    ones  = (ref_array*0+1).where(mask)
    dz = ones*dz_star*(1+(eta/h))
    return dz


def reconstruct_hrho_trans(u, v, wt, rho_dzt, grid, rho, reconstruct_w_from_cont=False):
    """Reconstruct thickness weighted mass transport in x,y,z direction.
    Units: uhrho_et/vhrho_nt [kg * s^-1 * m^-1]
           wrhot [kg * s^-1 * m^-2]"""
    rho_dzu = grid_shift(grid_shift(min_at_u(rho_dzt), 'X', grid), 'Y', grid)
    uhrho_et = remap_u_2_et((u * rho_dzu), grid)
    vhrho_nt = remap_v_2_nt((v * rho_dzu), grid)

    # I need to build in a routine to get w from cont
    if reconstruct_w_from_cont:
        raise RuntimeError('Not implemented')
#         wrhot = grid_shift((- uhrho_et - vhrho_nt), 'Z', grid)
    else:
        wrhot = wt * rho
    return uhrho_et, vhrho_nt, wrhot

def approximate_transport_op(uflux, vflux, wflux, tracer, grid, boundary='extend'):
    """Approximate the MOM5 transport operator. Fluxes are given as total tracer mass flux across x,y,z face
    Units: [tracer_units * kg * s^-1]"""
    tracer_xflux_adv = grid.interp(tracer, 'X', boundary=boundary) * uflux * grid._ds.dyte
    tracer_yflux_adv = grid.interp(tracer, 'Y', boundary=boundary) * vflux * grid._ds.dxtn
    tracer_zflux_adv = grid.interp(tracer, 'Z', boundary=boundary) * wflux * grid._ds.area_t
    return tracer_xflux_adv, tracer_yflux_adv, tracer_zflux_adv

def horizontal_t_cell_div(u, v, w, grid, boundary='extend'):
    """Computes a simple divergence operator and total change in mass/tracer
    Units:  div_u/div_v/div_w [tracer_units * kg * s^-1 * m^-1]
    """
    div_u = -grid.diff(u, 'X', boundary=boundary)
    div_v = -grid.diff(v, 'Y', boundary=boundary)
    div_w = grid.diff(w, 'Z', boundary=boundary)
    return div_u, div_v, div_w

def t_cell_tendency(uflux, vflux, wflux, tracer, grid):
    """converts thicknesswighted (horizontal) mass trans"""

    tracer_xflux_adv, tracer_yflux_adv, tracer_zflux_adv = \
        approximate_transport_op(uflux, vflux, wflux, tracer, grid)

    uf_div, vf_div, wf_div = horizontal_t_cell_div(tracer_xflux_adv,
                                                   tracer_yflux_adv,
                                                   tracer_zflux_adv,
                                                   grid)

    return (uf_div + vf_div + wf_div) / grid._ds.area_t

def grid_shift(da, axis, grid, boundary='extend'):
    ref = grid.interp(da, axis, boundary=boundary)
    return xr.DataArray(da.data, coords=ref.coords, dims=ref.coords)

# Reconstruct dzu (minimum of sourrounding grid cells)
def min_at_u(t_array, xdim='xt_ocean', ydim='yt_ocean'):
    ll = t_array
    lr = t_array.roll(**{xdim:-1})
    ul = t_array.roll(**{ydim:-1})
    ur = t_array.roll(**{xdim:-1, ydim:-1})
    u_min = xr.ufuncs.minimum(ul,ur)
    l_min = xr.ufuncs.minimum(ll,lr)
    return xr.ufuncs.minimum(u_min, l_min)

def remap_u_2_et(u, grid, boundary='extend'):
    dyu = grid._ds.dyu
    dyte = grid._ds.dyte
    u_on_et = grid.interp((u * dyu), 'Y', boundary=boundary) / dyte
    return u_on_et

def remap_v_2_nt(v, grid, boundary='extend'):
    dxu = grid._ds.dxu
    dxtn = grid._ds.dxtn
    v_on_nt = grid.interp((v * dxu), 'X', boundary=boundary) / dxtn
    return v_on_nt
