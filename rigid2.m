function [im2 delta] = rigid2(im1,im2,n)
%[im2 delta] = rigid2(im1,im2,n)
% aligns im2 with im1 using 2D rigid body registration
% -accepts 2D image, or stack of 2d images [nx ny ns]
% -n is the number of histogram bins (max 256)
% -im2 is returned registered using interp2
% -delta is the shifts [dx dy zrot]
%
% Ref: Lu et al (doi.org/10.1016/j.compmedimag.2007.12.001)
%
%% handle inputs
if ~isreal(im1) || ~isreal(im2) || ~isequal(size(im1),size(im2)) || ndims(im1)<2
    error('im1 and im2 must be real images of the same size.');
end

% Terrel-Scott rule (max 256)
if ~exist('n','var') || isempty(n)
    n = min(ceil((2*numel(im1))^(1/3)),256); 
elseif numel(n)~=1 || n<1 || n>256 || mod(n,1)
    error('number of histogram bins invalid.');
end

%% convert to n bins
uim1 = im2uint8(im1,n);
uim2 = im2uint8(im2,n);

% faster on gpu
try
    uim1 = gpuArray(uim1);
    uim2 = gpuArray(uim2);
end

%% calcuate transforms: [dx dy zrot]
opts = optimoptions('fminunc','SpecifyObjectiveGradient',true,'TolFun',1e-4,'display','iter');

cost = @(delta)hpv(uim1,uim2,delta,n);

delta = fminunc(cost,[0 0 0],opts)

%% apply transforms
[nx ny ns] = size(im1);

% get coordinates of im2
[~,~,x2,y2] = cost(delta);

% wrap edges and use 1-based indexing
x2 = mod(x2,nx)+1;
y2 = mod(y2,ny)+1;

% interpolate im2 to im1
[S L] = bounds(im2(:));
for s = 1:ns
    tmp = cast(im2(:,:,s),'like',x2); % cast to gpu/double
    tmp = interp2(tmp,y2,x2,'cubic'); % 'spline' fails on gpu
    im2(:,:,s) = reshape(tmp,nx,ny); % preserve original type
end
im2 = min(max(im2,S),L); % preserve bounds (e.g. nonnegative)

%% mutual information by joint histogram estimation (hpv)
function [fval grad x2 y2] = hpv(im1,im2,delta,n)

[nx ny ns] = size(im1);

if isa(im1,'gpuArray')
    nx = gpuArray(nx);
    ny = gpuArray(ny);
    ns = gpuArray(ns);
end

% joint histogram and partial derivatives [dx dy dzrot]
h = zeros(n,n,4,'like',nx);

% coordinates of im1 (centered at 0 0)
[x1 y1] = ndgrid(-nx/2:nx/2-1,-ny/2:ny/2-1);

% coordinates of im2 (centered at nx/2 ny/2)
sindzrot = sin(delta(3) * pi / 180);
cosdzrot = cos(delta(3) * pi / 180);
x2 = nx/2 + delta(1) + x1*cosdzrot - y1*sindzrot;
y2 = ny/2 + delta(2) + x1*sindzrot + y1*cosdzrot;

% early return for coordinates only
if nargout>2
    fval = []; grad = []; return;
end

% vectorize slices
im1 = reshape(im1,nx*ny,ns);
im2 = reshape(im2,nx*ny,ns);

% convolution with 4x4 Hann window
for i = -1:2
    for j = -1:2

        ix = floor(x2)+i; % convolution x-index
        iy = floor(y2)+j; % convolution y-index

        dx = x2-ix;
        dy = y2-iy;

        ix = mod(ix,nx);
        iy = mod(iy,ny);

        % histogram indices
        tmp = im2(1+ix+iy*nx,:); 
        ind = 1+[im1(:) tmp(:)]; clear ix iy tmp
        
        % function (k=1) and partial derivatives (k=2-4)            
        cosdx = cospi(dx/2); sindx = sinpi(dx/2); clear dx
        cosdy = cospi(dy/2); sindy = sinpi(dy/2); clear dy

        for k = 1:4
            switch(k)
                case 1; f(:,:,1) = (1+cosdx).*(1+cosdy);  % f
                case 2; f(:,:,2) =-(1+cosdy).*sindx*pi/2; % df/dx
                case 3; f(:,:,3) =-(1+cosdx).*sindy*pi/2; % df/dy
                case 4; f(:,:,4) =-f(:,:,2).*(x1*sindzrot+y1*cosdzrot)*pi/180 ...
                                  +f(:,:,3).*(x1*cosdzrot-y1*sindzrot)*pi/180; % df/dzrot            
            end

            tmp = repmat(reshape(f(:,:,k),[],1),ns,1);
            h(:,:,k) = h(:,:,k) + accumarray(ind,tmp,[n n]); clear tmp
        end

    end
end

pA = sum(h,2);
pB = sum(h,1);

h = reshape(h,[],4);
pA = reshape(pA,[],4);
pB = reshape(pB,[],4);

% mutual information
HA = plogp(pA(:,1));
HB = plogp(pB(:,1));
HAB = plogp(h(:,1));
fval = gather(HA+HB-HAB);

% partial derivatives
HA = plogdp(pA(:,1),pA(:,2:4));
HB = plogdp(pB(:,1),pB(:,2:4));
HAB = plogdp(h(:,1),h(:,2:4));
grad = gather(HA+HB-HAB);
grad = reshape(grad,size(delta));

%% perform log sums without NaN or Inf values
function s = plogp(p)
p(p<=0) = 1;
s = sum(p.*log(p));

function s = plogdp(p,dp)
p(p<=0) = exp(-1);
s = sum((1+log(p)).*dp);

%% convert image to uint8 from 0 to n-1
function im = im2uint8(im,n)
if ~isfloat(im)
    im = double(im);
end
im = im-min(im(:));
im = im/max(im(:));
im = uint8(im*(n-1));
