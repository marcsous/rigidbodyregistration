function [im2 delta] = rigid3(im1,im2,voxel)
%[im2 delta] = rigid3(im1,im2,voxel)
% aligns im2 with im1 using rigid body registration
%  -accepts 3d images or stack of 3d images [nx ny nz (ns)]
%  -voxel (optional) is the voxel dimensions [dx dy dz]
%  -im2 is returned registered using interp3
%  -delta is the shifts [dx dy dz xrot yrot zrot]
%
% Ref: Lu et al (doi.org/10.1016/j.compmedimag.2007.12.001)
%
%% setup
if ~isreal(im1) || ~isreal(im2) || nnz(~isfinite(im1)) || nnz(~isfinite(im2))
    error('im1 and im2 must be finite and real.');
end
if ndims(im1)<3 || ndims(im2)<3
    error('im1 and im2 must be 3d arrays.');
end
if ~exist('voxel','var') || isempty(voxel)
    voxel = [1 1 1];
elseif numel(voxel)~=3 || ~isreal(voxel) || nnz(voxel<=0) || nnz(~isfinite(voxel))
    error('voxel must be a real 3-element vector.');
end

% isotropic resolution unit used for registration 
unit = mean(voxel); % 1.0; 

%% input arrays at unit isotropic resolution for registration
try
    uim1 = gpuArray(im1);
    uim2 = gpuArray(im2);
catch
    uim1 = im1;
    uim2 = im2;
end

% interpolate to unit isotropic resolution
for dim = 1:3
    N = round(size(im2,dim) * voxel(dim) / unit);
    uim1 = interpft(uim1,N,dim);
    uim2 = interpft(uim2,N,dim);
end

%% convert to n bins - Terrel-Scott rule (max 256)
n = min(ceil((2*numel(uim2))^(1/3)),256);

uim1 = im2uint8(uim1,n);
uim2 = im2uint8(uim2,n);

%% calcuate rigid body parameters: [dx dy dz xrot yrot zrot]
t = tic;

opts = optimoptions('fminunc','SpecifyObjectiveGradient',true,'display','off');

cost = @(delta)hpv(uim1,uim2,delta,n,[1 1 1]);

delta = fminunc(cost,[0 0 0 0 0 0],opts);

fprintf('%s: %+.2f %+.2f %+.2f %+.2f %+.2f %+.2f (%.1f sec)\n',mfilename,delta,toc(t));

%% apply the transform to the native grid 

% convert shifts from isotropic to native voxels
delta(1:3) = delta(1:3) .* unit;

[nx ny nz ns] = size(im2);

if isa(im2,'gpuArray')
    nx = gpuArray(nx);
    ny = gpuArray(ny);
    nz = gpuArray(nz);
end
[x2 y2 z2] = get_coords([nx ny nz],voxel,delta);

% wrap edges (1-based indexing)
x2 = reshape(mod(x2,nx)+1,[nx ny nz]);
y2 = reshape(mod(y2,ny)+1,[nx ny nz]);
z2 = reshape(mod(z2,nz)+1,[nx ny nz]);

% preserve bounds (e.g. nonnegative)
[S L] = bounds(reshape(im2,[],1));

% interpolate 
for s = 1:ns
    % cubic may fail on gpu, linear looks lowpass filtered
    try
        im2(:,:,:,s) = interp3(im2(:,:,:,s),y2,x2,z2,'cubic',0);
    catch
        im2(:,:,:,s) = interp3(im2(:,:,:,s),y2,x2,z2,'nearest',0);
    end
end
im2 = min(max(im2,S),L);

%% rigid body coordinates on a grid with spacing of "voxel"
function [x2 y2 z2 PDx PDy PDz] = get_coords(sz,voxel,delta)

nx = sz(1); ny = sz(2); nz = sz(3);

% coordinates of im1 (centered at 0 0 0)
[x1 y1 z1] = ndgrid(-nx/2:nx/2-1,-ny/2:ny/2-1,-nz/2:nz/2-1);

% convert to matrix form
P = [x1(:) y1(:) z1(:)] .* reshape(voxel,1,[]);

% ease notation
s4 = sind(delta(4)); c4 = cosd(delta(4));
s5 = sind(delta(5)); c5 = cosd(delta(5));
s6 = sind(delta(6)); c6 = cosd(delta(6));

% rotation matrices
Rx = [1,0,0; 0,c5,s5; 0,-s5,c5]; % about x-axis
Ry = [c4,0,-s4; 0,1,0; s4,0,c4]; % about y-axis
Rz = [c6,s6,0; -s6,c6,0; 0,0,1]; % about z-axis
Rxyz = Rx * Ry * Rz;

% coordinates of im2 (centered at nx/2 ny/2 nz/2)
x2 = (P*Rxyz(:,1) + delta(1)) / voxel(1) + nx/2;
y2 = (P*Rxyz(:,2) + delta(2)) / voxel(2) + ny/2;
z2 = (P*Rxyz(:,3) + delta(3)) / voxel(3) + nz/2;

% partial derivatives of Rxyz wrt delta(4-6)
PDx = P*[        -c6*s4,         -s4*s6,   -c4;
               c4*c6*s5,       c4*s5*s6,-s4*s5;
               c4*c5*c6,       c4*c5*s6,-c5*s4];

PDy = P*[             0,              0,     0;
         c5*c6*s4+s5*s6,-c6*s5+c5*s4*s6, c4*c5;
         c5*s6-c6*s4*s5,-c5*c6-s4*s5*s6,-c4*s5];

PDz = P*[        -c4*s6,          c4*c6,     0;
        -c5*c6-s4*s5*s6,-c5*s6+c6*s4*s5,     0;
         c6*s5-c5*s4*s6, c5*c6*s4+s5*s6,     0];

%% mutual information by joint histogram estimation (hpv)
function [fval grad] = hpv(im1,im2,delta,n,voxel)

[nx ny nz ns] = size(im1);

nx = single(nx);
ny = single(ny);
nz = single(nz);
ns = single(ns);

if isa(im1,'gpuArray')
    nx = gpuArray(nx);
    ny = gpuArray(ny);
    nz = gpuArray(nz);
end

[x2 y2 z2 PDx PDy PDz] = get_coords([nx ny nz],voxel,delta);

% vectorize slab groups
im1 = reshape(im1,nx*ny*nz,ns);
im2 = reshape(im2,nx*ny*nz,ns);

% joint histogram and partial derivatives [dx dy dz dxrot dyrot dzrot]
h = zeros(n,n,7,'like',nx);

% convolution with 4x4 Hann window
for i = -1:2
    for j = -1:2
        for k = -1:2

            ix = floor(x2)+i; % convolution x-index
            iy = floor(y2)+j; % convolution y-index
            iz = floor(z2)+k; % convolution z-index

            dx = x2-ix;
            dy = y2-iy;
            dz = z2-iz;

            ix = mod(ix,nx);
            iy = mod(iy,ny);
            iz = mod(iz,nz);

            % histogram indices
            index = im2(1+ix+iy*nx+iz*nx*ny,:); clear ix iy iz
            index = 1 + [im1(:) index(:)];

            % function (m=1) and partial derivatives (m=2-7)
            cosdx = cospi(dx/2); sindx = sinpi(dx/2); clear dx
            cosdy = cospi(dy/2); sindy = sinpi(dy/2); clear dy
            cosdz = cospi(dz/2); sindz = sinpi(dz/2); clear dz

            for m = 1:7

                switch(m)
                    case 1; f = (1+cosdx).*(1+cosdy).*(1+cosdz);               % f
                    case 2; f = (0-sindx).*(1+cosdy).*(1+cosdz)*pi/2/voxel(1); % df/dx
                    case 3; f = (1+cosdx).*(0-sindy).*(1+cosdz)*pi/2/voxel(2); % df/dy
                    case 4; f = (1+cosdx).*(1+cosdy).*(0-sindz)*pi/2/voxel(3); % df/dz
                    case 5; f = dot(F,PDx,2)*pi/180;                           % df/dxrot
                    case 6; f = dot(F,PDy,2)*pi/180;                           % df/dyrot
                    case 7; f = dot(F,PDz,2)*pi/180;                           % df/dzrot
                end

                % help with out-of-memory issues
                if m==2; F(:,1) = f; clear sindx; end
                if m==3; F(:,2) = f; clear sindy cosdz; end
                if m==4; F(:,3) = f; clear sindz cosdx cosdy; end
    
                if ns>1; f = repmat(f,ns,1); end
                h(:,:,m) = h(:,:,m) + accumarray(index,f,[n n]);

            end

        end
    end
end

pA = sum(h,2);
pB = sum(h,1);

h = reshape(h,[],7);
pA = reshape(pA,[],7);
pB = reshape(pB,[],7);

% mutual information
HA = plogp(pA(:,1));
HB = plogp(pB(:,1));
HAB = plogp(h(:,1));
fval = gather(double(HA+HB-HAB));

% partial derivatives
HA = dplogp(pA(:,2:7),pA(:,1));
HB = dplogp(pB(:,2:7),pB(:,1));
HAB = dplogp(h(:,2:7),h(:,1));
grad = gather(double(HA+HB-HAB));
grad = reshape(grad,size(delta));

%% perform log sums without NaN or Inf values
function s = plogp(p)
p(p<=0) = 1;
s = sum(p.*log(p));

function s = dplogp(dp,p)
p(p<=0) = exp(-1);
s = sum(dp.*(log(p)+1));

%% convert image to uint8 from 0 to n-1
function im = im2uint8(im,n)
if ~isfloat(im)
    im = double(im);
end
im = im-min(im(:));
im = im/max(im(:));
im = uint8(im*(n-1));
