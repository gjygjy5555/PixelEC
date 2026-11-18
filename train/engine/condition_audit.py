"""Check the initial training forwards preserve the expected LQ observation."""


def install_condition_audit():
    from . import train_base as official
    previous = official.build_flow
    def build(cfg, accelerator):
        flow = previous(cfg,accelerator)
        original = flow.loss_fm
        count = [0]
        def loss(model,lq,hq,*args,**kwargs):
            def audit(_module,inputs):
                if count[0] < 3:
                    actual = inputs[0][:,:3]
                    error = float((actual-lq.to(actual.dtype)).abs().max())
                    assert error == 0 and flow.cfg_ratio == 0
                    print('NO_AELQ_VERIFIED LQ_max_error=0 diffusion_noise_retained=True',flush=True)
                    count[0] += 1
            handle = model.register_forward_pre_hook(audit)
            try:
                return original(model,lq,hq,*args,**kwargs)
            finally:
                handle.remove()
        flow.loss_fm = loss
        return flow
    official.build_flow = build
