function [im2 delta] = rigid2(im1,im2,voxel)
%[im2 delta] = rigid2(im1,im2,n)
% aligns im2 with im1 using 2d rigid body registration
%  -accepts 2D image, or stack of 2d images [nx ny (ns)]
%  -voxel (optional) is the voxel dimensions (mm) [dx dy]
%  -im2 is returned registered using interp2
%  -delta is the shifts [dx dy zrot]
%
% Ref: Lu et al (doi.org/10.1016/j.compmedimag.2007.12.001)
%
%% handle inputs
if ~isreal(im1) || ~isreal(im2) || nnz(~isfinite(im1)) || nnz(~isfinite(im2))
    error('im1 and im2 must be finite and real.');
end
if ~exist('voxel','var') || isempty(voxel)
    voxel = ones(2,1);
elseif numel(voxel)~=2 || ~isreal(voxel) || any(voxel<=0) || any(~isfinite(voxel))
    error('voxel must be a real 2-element vector.');
end

% isotropic resolution unit used for registration 
unit = mean(voxel); % 1.0; 

%% copy of input arrays for registration
try
    uim1 = gpuArray(im1);
    uim2 = gpuArray(im2);
catch
    uim1 = im1;
    uim2 = im2;
end

% interpolate to "1x1x1" resolution
for dim = 1:2
    N = round(size(im2,dim) * voxel(dim) / unit);
    uim1 = interpft(uim1,N,dim);
    uim2 = interpft(uim2,N,dim);
end

%% convert to n bins - Terrel-Scott rule (max 256)
n = min(ceil((2*numel(uim2))^(1/3)),256); 

uim1 = im2uint8(uim1,n);
uim2 = im2uint8(uim2,n);

%% calcuate transforms: [dx dy zrot]
t = tic;

opts = optimoptions('fminunc','SpecifyObjectiveGradient',true,'display','off');

cost = @(delta)hpv(uim1,uim2,delta,n,[1 1]);

delta = fminunc(cost,[0 0 0],opts);

fprintf('%s: %+.2f %+.2f %+.2f (%.1f sec)\n',mfilename,delta,toc(t));

%% isotropic copies of input arrays for interpolation
try
    uim2 = gpuArray(im2);
catch
    uim2 = im2;
end

% interpolate to highest resolution
for dim = 1:2
    N = round(size(im2,dim) * voxel(dim) / min(voxel));
    uim2 = interpft(uim2,N,dim);
    delta(dim) = delta(dim) * unit / min(voxel);
end

%% apply the transform to the native grid 

% convert shifts from isotropic to native voxels
delta(1:2) = delta(1:2) .* unit;

[nx ny ns] = size(im2);

if isa(im2,'gpuArray')
    nx = gpuArray(nx);
    ny = gpuArray(ny);
end
[x2 y2] = get_coords([nx ny],voxel,delta);

% wrap edges (1-based indexing)
x2 = reshape(mod(x2,nx)+1,[nx ny]);
y2 = reshape(mod(y2,ny)+1,[nx ny]);

% preserve bounds (e.g. nonnegative)
[S L] = bounds(reshape(im2,[],1));

% interpolate 
for s = 1:ns
    % cubic may fail on gpu, linear looks lowpass filtered
    try
        im2(:,:,s) = interp2(im2(:,:,s),y2,x2,'cubic',0);
    catch
        im2(:,:,s) = interp2(im2(:,:,s),y2,x2,'nearest',0);
    end
end
im2 = min(max(im2,S),L);

%% rigid body coordinates on a grid with spacing of "voxel"
function [x2 y2 PDz] = get_coords(sz,voxel,delta)

nx = sz(1); ny = sz(2);

% coordinates of im1 (centered at 0 0)
[x1 y1] = ndgrid(-nx/2:nx/2-1,-ny/2:ny/2-1);

% convert to matrix form
P = [x1(:) y1(:)] .* reshape(voxel,1,[]);

% ease notation
sindzrot = sind(delta(3));
cosdzrot = cosd(delta(3));

% rotation matrix about z-axis
Rz = [+cosdzrot,+sindzrot;-sindzrot,+cosdzrot];

% coordinates of im2 (centered at nx/2 ny/2)
x2 = (P*Rz(:,1) + delta(1)) / voxel(1) + nx/2;
y2 = (P*Rz(:,2) + delta(2)) / voxel(2) + ny/2;

% partial derivative of Rz wrt delta(3)
PDz = P*[-sindzrot,+cosdzrot;+cosdzrot,-sindzrot];

%% mutual information by joint histogram estimation (hpv)
function [fval grad] = hpv(im1,im2,delta,n,voxel)

[nx ny ns] = size(im1);

nx = single(nx);
ny = single(ny);
ns = single(ns);

if isa(im1,'gpuArray')
    nx = gpuArray(nx);
    ny = gpuArray(ny);
    ns = gpuArray(ns);
end

[x2 y2 PDz] = get_coords([nx ny],voxel,delta);

% vectorize slices
im1 = reshape(im1,nx*ny,ns);
im2 = reshape(im2,nx*ny,ns);

% joint histogram and partial derivatives [dx dy dzrot]
h = zeros(n,n,4,'like',nx);

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
        index = im2(1+ix+iy*nx,:); clear ix iy
        index = 1 + [im1(:) index(:)]; 
        
        % function (m=1) and partial derivatives (m=2-4)            
        cosdx = cospi(dx/2); sindx = sinpi(dx/2); clear dx
        cosdy = cospi(dy/2); sindy = sinpi(dy/2); clear dy

        for m = 1:4
            switch(m)
                case 1; f = (1+cosdx).*(1+cosdy);               % f
                case 2; f = (0-sindx).*(1+cosdy)*pi/2/voxel(1); % df/dx
                case 3; f = (1+cosdx).*(0-sindy)*pi/2/voxel(2); % df/dy
                case 4; f = dot(F,PDz,2)*pi/180;                % df/dzrot
            end

            % help with out-of-memory issues
            if m==2; F(:,1) = f; clear sindx cosdy; end
            if m==3; F(:,2) = f; clear cosdx sindy; end

            if ns>1; f = repmat(f,ns,1); end
            h(:,:,m) = h(:,:,m) + accumarray(index,f,[n n]);

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
fval = gather(double(HA+HB-HAB));

% partial derivatives
HA = plogdp(pA(:,1),pA(:,2:4));
HB = plogdp(pB(:,1),pB(:,2:4));
HAB = plogdp(h(:,1),h(:,2:4));
grad = gather(double(HA+HB-HAB));
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
