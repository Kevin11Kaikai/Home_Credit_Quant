"""Full-width synthetic throughput probe, NOT a predictive experiment."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from masamlp.models import build_model
from vendor.excel import ExcelFormer
HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    schema=json.loads((HERE/'../champion_features_20261014/derived/cat/schema.json').read_text())
    n_num,n_cat=len(schema['numeric']),len(schema['categorical'])
    results=[]
    for name in ['denselight','excel','s6e7']:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.manual_seed(42)
        if name=='s6e7':
            num=n_num+(n_num+n_cat)
            model=build_model('ft_transformer',dict(d_block=128,n_blocks=2,attention_n_heads=8,n_frequencies=24,sigma=.1),n_num=num,cat_cardinalities=[50]*n_cat,out_dim=1,num_embedding='plr-lite').cuda()
            x=torch.randn(128,num,device='cuda');c=torch.randint(0,50,(128,n_cat),device='cuda');batch=16
        elif name=='excel':
            num=n_num+n_cat+100 # conservative probe; real dedup flag count measured during prep
            model=ExcelFormer(d_numerical=num,categories=None,token_bias=True,n_layers=3,d_token=128,n_heads=8,attention_dropout=.3,ffn_dropout=0.,residual_dropout=0.,prenormalization=True,kv_compression=None,kv_compression_sharing=None,d_out=1,init_scale=.01).cuda()
            x=torch.randn(128,num,device='cuda');c=None;batch=8
        else:
            from lightautoml.ml_algo.torch_based.nn_models import DenseLightModel
            num=n_num+n_cat
            model=DenseLightModel(n_in=num).cuda();x=torch.randn(1024,num,device='cuda');c=None;batch=256
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        y=torch.randint(0,2,(len(x),),device='cuda').float()
        timings=[]
        for step in range(5):
            torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=model(x[:batch],c[:batch]) if name=='s6e7' else (model(x[:batch],None) if name=='excel' else model(x[:batch]))
                loss=F.binary_cross_entropy_with_logits(z.reshape(-1).float(),y[:batch])
            assert torch.isfinite(loss);loss.backward();opt.step();torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
        rate=batch/float(np.median(timings[1:]))
        row=dict(model=name,synthetic_only=True,microbatch=batch,numerical_inputs=num,categorical_inputs=n_cat if name=='s6e7' else 0,samples_per_second=rate,seconds_per_epoch_267645=267645/rate,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,steps_seconds=timings,gpu=torch.cuda.get_device_name(0))
        results.append(row);print(json.dumps(row),flush=True)
        del model,opt,x,c,y,z,loss;gc.collect()
    (HERE/'compute_probe.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
