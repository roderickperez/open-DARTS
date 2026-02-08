import numpy as np
from dartsflash.components import CompData
from dartsflash.dartsflash import DARTSFlash, EoS, FlashParams


class IAPWS(DARTSFlash):
    def __init__(self, iapws_ideal: bool = True, ice_phase: bool = False):
        super().__init__(comp_data=CompData(components=["H2O"]))

        # Add IAPWS-95 EoS
        iapws = self.get_eos["IAPWS"](self.comp_data, iapws_ideal=iapws_ideal)
        self.add_eos("IAPWS", iapws, root_order=[EoS.RootFlag.MAX, EoS.RootFlag.MIN])

        # Add IAPWS-ice EoS, used only if in ice conditions
        self.eos_order = ["IAPWS", "Ice"] if ice_phase else ["IAPWS"]
        if ice_phase:
            ice = self.get_eos["IAPWSIce"](self.comp_data, iapws_ideal=iapws_ideal)
            self.add_eos("Ice", ice)

    def init_flash(self, flash_type: DARTSFlash.FlashType, pxflash_type: FlashParams.PXFlashType = FlashParams.BRENT,
                   f_tol: float = 1e-6, t_tol: float = 1e-2, t_min: float = 100., t_max: float = 1000., t_init: float = 300.,
                   verbose: bool = False):
        # Set flash parameters in FlashParams
        super().init_flash(flash_type=flash_type, eos_order=self.eos_order, pxflash_type=pxflash_type,
                           t_min=t_min, t_max=t_max, f_tol=f_tol, t_tol=t_tol, verbose=verbose)


class VL(DARTSFlash):
    def set_vl_eos(self, vl_eos, root_order: list = None, trial_comps: list = None, rich_phase_order: list = None,
                   stability_tol: float = 1e-20, switch_tol: float = 1e-3, max_iter: int = 100, use_gmix: bool = False):
        # Set default parameters
        trial_comps = trial_comps if trial_comps is not None else [i for i in range(self.nc)]
        root_order = [EoS.RootFlag.MAX, EoS.RootFlag.MIN] if root_order is None else root_order
        rich_phase_order = rich_phase_order if rich_phase_order is not None else []

        self.add_eos("VL", self.get_eos[vl_eos](self.comp_data), trial_comps=trial_comps,
                     root_order=root_order, stability_tol=stability_tol, switch_tol=switch_tol,
                     max_iter=max_iter, rich_phase_order=rich_phase_order, use_gmix=use_gmix)

    def init_flash(self, flash_type: DARTSFlash.FlashType, min_z: float = 1e-13, y_pure: float = 0.9,
                   tpd_tol: float = 1e-8, tpd_1p_tol: float = 1e-4, tpd_close_to_boundary: float = 1e-3, comp_tol: float = 1e-4,
                   rr2_tol: float = 1e-12, rrn_tol: float = 1e-14, rr_max_iter: int = 100, rr_line_iter: int = 10,
                   split_tol: float = 1e-15, split_switch_tol: float = 1e-15, split_switch_diff: float = 2.,
                   split_line_tol: float = 1e-8, split_negative_flash_tol: float = 1e-4,
                   split_max_iter: int = 500, split_line_iter = 100, split_negative_flash_iter = 500,
                   split_variables: FlashParams.SplitVars = FlashParams.SplitVars.lnK,
                   stability_variables: FlashParams.StabilityVars = FlashParams.StabilityVars.alpha,
                   pxflash_type: FlashParams.PXFlashType = FlashParams.BRENT, f_tol: float = 1e-6, t_tol: float = 1e-1,
                   t_min: float = 100., t_max: float = 1000., t_init: float = 300., verbose: bool = False,
                   vl_eos_name: str = "", light_comp_idx: int = -1, nf_initial_guess: list = None):
        assert "VL" in self.eos.keys(), "Please specify VL EoS"

        # Call base init_flash()
        super().init_flash(flash_type=flash_type, eos_order=["VL"], min_z=min_z, y_pure=y_pure,
                           tpd_tol=tpd_tol, tpd_1p_tol=tpd_1p_tol, tpd_close_to_boundary=tpd_close_to_boundary,
                           comp_tol=comp_tol, rr2_tol=rr2_tol, rrn_tol=rrn_tol, rr_max_iter=rr_max_iter,
                           rr_line_iter=rr_line_iter, split_tol=split_tol, split_switch_tol=split_switch_tol, split_switch_diff=split_switch_diff,
                           split_line_tol=split_line_tol, split_negative_flash_tol=split_negative_flash_tol,
                           split_max_iter=split_max_iter, split_line_iter=split_line_iter,
                           split_negative_flash_iter=split_negative_flash_iter,
                           split_variables=split_variables, stability_variables=stability_variables,
                           pxflash_type=pxflash_type, f_tol=f_tol, t_tol=t_tol, t_min=t_min, t_max=t_max, t_init=t_init, 
                           vl_eos_name=vl_eos_name, light_comp_idx=light_comp_idx,
                           verbose=verbose, nf_initial_guess=nf_initial_guess)


class VLAq(DARTSFlash):
    def __init__(self, comp_data: CompData, mixture_name: str = None, hybrid: bool = True):
        super().__init__(comp_data, mixture_name=mixture_name)
        self.hybrid = hybrid
        self.H2O_idx = comp_data.components.index("H2O")

    def set_vl_eos(self, vl_eos: str, trial_comps: list = None, stability_tol: float = 1e-20, switch_tol: float = 1e-3,
                   max_iter: int = 100, root_order: list = None, rich_phase_order: list = None, use_gmix: bool = False):
        # Set default parameters
        trial_comps = trial_comps if trial_comps is not None else [i for i in range(self.nc)]
        root_order = root_order if root_order is not None else [EoS.RootFlag.MAX, EoS.RootFlag.MIN]
        rich_phase_order = rich_phase_order if rich_phase_order is not None else []

        # Set preferred roots for hybrid-EoS approach
        preferred_roots = [(self.H2O_idx, 0.8, EoS.MAX)] if self.hybrid else None

        # Add VL EoS object from get_eos()
        self.add_eos("VL", self.get_eos[vl_eos](comp_data=self.comp_data), trial_comps=trial_comps,
                     stability_tol=stability_tol, switch_tol=switch_tol, max_iter=max_iter, use_gmix=use_gmix,
                     preferred_roots=preferred_roots, root_order=root_order, rich_phase_order=rich_phase_order)

    def set_aq_eos(self, aq_eos: str, stability_tol: float = 1e-20, switch_tol: float = 1e-20, max_iter: int = 10,
                   use_gmix: bool = False):
        # Add AQEoS object from get_eos()
        self.add_eos("Aq", self.get_eos[aq_eos](comp_data=self.comp_data),
                     eos_range={self.H2O_idx: [0.6, 1.]}, trial_comps=[self.H2O_idx],
                     stability_tol=stability_tol, switch_tol=switch_tol, max_iter=max_iter, use_gmix=use_gmix)

    def init_flash(self, flash_type: DARTSFlash.FlashType, eos_order: list = None, min_z: float = 1e-13, y_pure: float = 0.9,
                   tpd_tol: float = 1e-8, tpd_1p_tol: float = 1e-4, tpd_close_to_boundary: float = 1e-3, comp_tol: float = 1e-4,
                   rr2_tol: float = 1e-12, rrn_tol: float = 1e-14, rr_max_iter: int = 100, rr_line_iter: int = 10,
                   split_tol: float = 1e-15, split_switch_tol: float = 1e-15, split_switch_diff: float = 2.,
                   split_line_tol: float = 1e-8, split_negative_flash_tol: float = 1e-4,
                   split_max_iter: int = 500, split_line_iter = 100, split_negative_flash_iter = 500,
                   split_variables: FlashParams.SplitVars = FlashParams.SplitVars.lnK,
                   stability_variables: FlashParams.StabilityVars = FlashParams.StabilityVars.alpha,
                   pxflash_type: FlashParams.PXFlashType = FlashParams.BRENT, f_tol: float = 1e-6, t_tol: float = 1e-1,
                   t_min: float = 100., t_max: float = 1000., t_init: float = 300., vl_eos_name: str = "", light_comp_idx: int = -1, 
                   verbose: bool = False, nf_initial_guess: list = None):
        assert "VL" in self.eos.keys(), "Please specify VL EoS"
        if self.hybrid:
            assert "Aq" in self.eos.keys(), "Please specify Aq EoS"

        # Call base init_flash()
        eos_order = (["VL", "Aq"] if self.hybrid else ["VL"]) if eos_order is None else eos_order
        super().init_flash(eos_order=eos_order, flash_type=flash_type, min_z=min_z, y_pure=y_pure,
                           tpd_tol=tpd_tol, tpd_1p_tol=tpd_1p_tol, tpd_close_to_boundary=tpd_close_to_boundary,
                           comp_tol=comp_tol, rr2_tol=rr2_tol, rrn_tol=rrn_tol, rr_max_iter=rr_max_iter,
                           rr_line_iter=rr_line_iter, split_tol=split_tol, split_switch_tol=split_switch_tol, split_switch_diff=split_switch_diff,
                           split_line_tol=split_line_tol, split_negative_flash_tol=split_negative_flash_tol,
                           split_max_iter=split_max_iter, split_line_iter=split_line_iter,
                           split_negative_flash_iter=split_negative_flash_iter,
                           split_variables=split_variables, stability_variables=stability_variables,
                           t_min=t_min, t_max=t_max, t_init=t_init, f_tol=f_tol, t_tol=t_tol, 
                           vl_eos_name=vl_eos_name, light_comp_idx=light_comp_idx,
                           verbose=verbose, nf_initial_guess=nf_initial_guess)


class VLAqH(VLAq):
    def set_h_eos(self, hydrate_type: str = None, stability_tol: float = 1e-20, switch_tol: float = 1e-1,
                  max_iter: int = 20, use_gmix: bool = False):
        self.hydrate_type = hydrate_type
        if hydrate_type is None:
            self.add_eos("sI", self.get_eos["Ballard"](comp_data=self.comp_data, hydrate_type="sI"),
                         trial_comps=[self.H2O_idx],
                         stability_tol=stability_tol, switch_tol=switch_tol, max_iter=max_iter, use_gmix=use_gmix)
            self.add_eos("sII", self.get_eos["Ballard"](comp_data=self.comp_data, hydrate_type="sII"),
                         trial_comps=[self.H2O_idx],
                         stability_tol=stability_tol, switch_tol=switch_tol, max_iter=max_iter, use_gmix=use_gmix)
            # self.add_eos("sH", self.get_eos["Ballard"](comp_data=self.comp_data, hydrate_type="sH"),
            #              trial_comps=[self.H2O_idx],
            #              stability_tol=stability_tol, switch_tol=switch_tol, max_iter=max_iter, use_gmix=use_gmix)
        else:
            self.add_eos(hydrate_type, self.get_eos["Ballard"](comp_data=self.comp_data, hydrate_type=hydrate_type),
                         trial_comps=[self.H2O_idx],
                         stability_tol=stability_tol, switch_tol=switch_tol, max_iter=max_iter, use_gmix=use_gmix)

    def init_flash(self, flash_type: DARTSFlash.FlashType, eos_order: list = None, min_z: float = 1e-13, y_pure: float = 0.9,
                   tpd_tol: float = 1e-8, tpd_1p_tol: float = 1e-4, tpd_close_to_boundary: float = 1e-3, comp_tol: float = 1e-4,
                   rr2_tol: float = 1e-12, rrn_tol: float = 1e-14, rr_max_iter: int = 100, rr_line_iter: int = 10,
                   split_tol: float = 1e-15, split_switch_tol: float = 1e-15, split_switch_diff: float = 2.,
                   split_line_tol: float = 1e-8, split_negative_flash_tol: float = 1e-4,
                   split_max_iter: int = 500, split_line_iter = 100, split_negative_flash_iter = 500,
                   split_variables: FlashParams.SplitVars = FlashParams.SplitVars.lnK,
                   stability_variables: FlashParams.StabilityVars = FlashParams.StabilityVars.alpha,
                   pxflash_type: FlashParams.PXFlashType = FlashParams.BRENT, f_tol: float = 1e-3, t_tol: float = 1e-1,
                   t_min: float = 100., t_max: float = 1000., t_init: float = 300., 
                   vl_eos_name: str = "", light_comp_idx: int = -1, verbose: bool = False):
        assert "VL" in self.eos.keys(), "Please specify VL EoS"
        assert "Aq" in self.eos.keys(), "Please specify Aq EoS"
        assert (self.hydrate_type if self.hydrate_type is not None else "sI") in self.eos.keys(), "Please specify hydrate EoS"

        eos_order = ["VL", "Aq", "sI"] if eos_order is None else eos_order
        super().init_flash(flash_type=flash_type, eos_order=eos_order, min_z=min_z, y_pure=y_pure,
                           tpd_tol=tpd_tol, tpd_1p_tol=tpd_1p_tol, tpd_close_to_boundary=tpd_close_to_boundary,
                           comp_tol=comp_tol, rr2_tol=rr2_tol, rrn_tol=rrn_tol, rr_max_iter=rr_max_iter, rr_line_iter=rr_line_iter,
                           split_tol=split_tol, split_switch_tol=split_switch_tol, split_switch_diff=split_switch_diff,
                           split_line_tol=split_line_tol, split_negative_flash_tol=split_negative_flash_tol,
                           split_max_iter=split_max_iter, split_line_iter=split_line_iter, split_negative_flash_iter=split_negative_flash_iter,
                           split_variables=split_variables, stability_variables=stability_variables, pxflash_type=pxflash_type,
                           f_tol=f_tol, t_tol=t_tol, t_min=t_min, t_max=t_max, t_init=t_init, 
                           vl_eos_name=vl_eos_name, light_comp_idx=light_comp_idx, verbose=verbose)
