import sys, numpy as np
from PIL import Image
from scipy import ndimage

SRC='/root/.claude/uploads/5ef10e14-19c0-5fa2-a922-b7dcf50918c0/a6aeca0a-IMG_9844.jpeg'
K2,S2,K3,S3,OUT=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),sys.argv[5]
a=np.asarray(Image.open(SRC).convert('RGB')).astype(np.float64)
H,W,_=a.shape; R,G,B=a[:,:,0],a[:,:,1],a[:,:,2]

crimson=(G<105)&(R>110)&(B<140)
solid=ndimage.binary_fill_holes(ndimage.binary_closing(crimson,np.ones((5,5))))
near=ndimage.binary_dilation(solid,np.ones((9,9)))
white=(R>205)&((R-G)<32)&((R-B)<32)&near
gold=(R>150)&(G>100)&(B<G-15)&(R-B>45)&near
ink=ndimage.binary_opening(ndimage.binary_fill_holes(
        ndimage.binary_closing(solid|white|gold,np.ones((7,7)))),np.ones((5,5)))

LINES=[('line2',196,470, 95,1200, K2,S2),('line3',462,668, 250,1120, K3,S3)]
lab,n=ndimage.label(ndimage.binary_dilation(ink,np.ones((9,9))))
objs=ndimage.find_objects(lab); sizes=ndimage.sum(np.ones_like(lab),lab,range(1,n+1))
sel={ln[0]:np.zeros((H,W),bool) for ln in LINES}
for i in range(n):
    if sizes[i]<500: continue
    ys,xs=objs[i]; cy=(ys.start+ys.stop)/2; cx=(xs.start+xs.stop)/2
    for name,y0,y1,xa,xb,k,s in LINES:
        if y0<=cy<y1 and xa<=cx<=xb and xs.start>=xa-30 and xs.stop<=xb+30:
            sel[name]|=(lab==i+1); break
moved=np.zeros((H,W),bool)
for m in sel.values(): moved|=m

tight=ndimage.binary_dilation(moved,np.ones((9,9)))
FEATHER=34.0
dist=ndimage.distance_transform_edt(~tight)
wgt=np.clip(1.0-dist/FEATHER,0,1)          # 1 on the glyphs, fading out over 34px

low=a.copy()
for ci in range(3):
    ch=a[:,:,ci]; cur=np.where(tight,0,ch); wt=(~tight).astype(np.float64); est=ch
    for sig in (140,90,55,34,21,13,8):
        s_=ndimage.gaussian_filter(cur,sig); w=ndimage.gaussian_filter(wt,sig)
        est=s_/np.maximum(w,1e-8); cur=np.where(tight,est,ch); wt=np.ones_like(wt)
    low[:,:,ci]=est
filled=a*(1-wgt[...,None])+low*wgt[...,None]

res=filled.copy()
for name,y0,y1,xa,xb,k,s in LINES:
    m=sel[name]; yy,xx=np.nonzero(m)
    lcx=(xx.min()+xx.max())/2; lcy=(yy.min()+yy.max())/2
    sxs=s/k; sys_=s
    al=np.clip((ndimage.gaussian_filter(ndimage.binary_dilation(m,np.ones((21,21))).astype(np.float64),9.0)-0.16)/0.5,0,1)
    coef=(1/sxs,0,lcx*(1-1/sxs), 0,1/sys_,lcy*(1-1/sys_))
    lw=np.asarray(Image.fromarray(a.clip(0,255).astype(np.uint8)).transform((W,H),Image.AFFINE,coef,resample=Image.BICUBIC)).astype(np.float64)
    la=np.asarray(Image.fromarray((al*255).astype(np.uint8)).transform((W,H),Image.AFFINE,coef,resample=Image.BICUBIC)).astype(np.float64)/255
    res=res*(1-la[...,None])+lw*la[...,None]
Image.fromarray(res.clip(0,255).astype(np.uint8)).save(OUT)
print('wrote',OUT)
