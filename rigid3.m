function [im2 delta] = rigid3(im1,im2,n)
%[im2 delta] = rigid3(im1,im2,n)
% aligns im2 with im1 using rigid body registration
% -accepts 3d images or stack of 3d images [nx ny nz (ns)]
% -n is the number of histogram bins (max 256)
% -im2 is returned registered using interp3
% -delta is the shifts [dx dy dz xrot yrot zrot]
%
% Ref: Lu et al (doi.org/10.1016/j.compmedimag.2007.12.001)
%
%% handle inputs
if ~isreal(im1) || ~isreal(im2) || nnz(isfinite(im1)==0) || nnz(isfinite(im2)==0)
    error('im1 and im2 must be real valued.');
end
if ~isequal(size(im1),size(im2)) || ndims(im1)<3 || any(size(im1)<=1)
    error('im1 and im2 must be 3d arrays of the same size.');
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

%% calcuate rigid body parameters: [dx dy dz xrot yrot zrot]
opts = optimoptions('fminunc','SpecifyObjectiveGradient',true,'TolFun',1e-4,'display','iter');

cost = @(delta)hpv(uim1,uim2,delta,n);

delta = fminunc(cost,[0 0 0 0 0 0],opts)

%% apply the transforms
[nx ny nz ns] = size(im1);

% get coordinates of im2
[~,~,x2,y2,z2] = cost(delta);
clear cost uim1 uim2 opts

% wrap edges and use 1-based indexing
x2 = mod(x2,nx)+1;
y2 = mod(y2,ny)+1;
z2 = mod(z2,nz)+1;

% interpolate im2 to im1 - spline/cubic may fail on gpu
tmp = cast(im2,'like',x2);
for s = 1:ns
    try
        tmp(:,:,:,s) = interp3(tmp(:,:,:,s),y2,x2,z2,'cubic',0);
    catch
        tmp(:,:,:,s) = interp3(tmp(:,:,:,s),y2,x2,z2,'linear',0);
    end
end
tmp = cast(tmp,'like',im2);

% preserve bounds (nonnegative) and type
[S L] = bounds(im2(:));
im2 = min(max(tmp,S),L);

%% mutual information by joint histogram estimation (hpv)
function [fval grad x2 y2 z2] = hpv(im1,im2,delta,n)

[nx ny nz ns] = size(im1);

if isa(im1,'gpuArray')
    nx = gpuArray(nx);
    ny = gpuArray(ny);
    nz = gpuArray(nz);
end

% joint histogram and partial derivatives [dx dy dz dxrot dyrot dzrot]
h = zeros(n,n,7,'like',nx);

% coordinates of im1 (centered at 0 0 0)
[x1 y1 z1] = ndgrid(-nx/2:nx/2-1,-ny/2:ny/2-1,-nz/2:nz/2-1);

% convert to matrix form
P = cat(2,reshape(x1,[],1),reshape(y1,[],1),reshape(z1,[],1)); clear x1 y1 z1

% ease notation, convert to radians
s4 = sin(delta(4) * pi / 180);
s5 = sin(delta(5) * pi / 180);
s6 = sin(delta(6) * pi / 180);
c4 = cos(delta(4) * pi / 180);
c5 = cos(delta(5) * pi / 180);
c6 = cos(delta(6) * pi / 180);

% rotation matrices
Rx = [1,0,0;0,c5,s5;0,-s5,c5]; % about x-axis
Ry = [c4,0,-s4;0,1,0;s4,0,c4]; % about y-axis
Rz = [c6,s6,0;-s6,c6,0;0,0,1]; % about z-axis
Rxyz = Rx * Ry * Rz;

% partial derivatives of Rxyz wrt delta(4-6)
Dx = [        -c6*s4,         -s4*s6,   -c4;
            c4*c6*s5,       c4*s5*s6,-s4*s5;
            c4*c5*c6,       c4*c5*s6,-c5*s4];
Dy = [             0,              0,     0;
      c5*c6*s4+s5*s6,-c6*s5+c5*s4*s6, c4*c5;
      c5*s6-c6*s4*s5,-c5*c6-s4*s5*s6,-c4*s5];
Dz = [        -c4*s6,          c4*c6,     0;
     -c5*c6-s4*s5*s6,-c5*s6+c6*s4*s5,     0;
      c6*s5-c5*s4*s6, c5*c6*s4+s5*s6,     0];

% coordinates of im2 (centered at nx/2 ny/2 nz/2)
x2 = nx/2 + delta(1) + P * Rxyz(:,1);
y2 = ny/2 + delta(2) + P * Rxyz(:,2);
z2 = nz/2 + delta(3) + P * Rxyz(:,3);

% early return for coordinates only
if nargout>2
    x2 = reshape(x2,[nx ny nz]);
    y2 = reshape(y2,[nx ny nz]);
    z2 = reshape(z2,[nx ny nz]);
    fval = 0; grad = 0; return;
end

% vectorize slice groups
im1 = reshape(im1,[],ns);
im2 = reshape(im2,[],ns);

% convolution with 4x4 Hann window
for i = -1:2
    for j = -1:2
        for k = -1:2

            xi = floor(x2)+i; % convolution x-index
            yi = floor(y2)+j; % convolution y-index
            zi = floor(z2)+k; % convolution z-index

            dx = x2-xi;
            dy = y2-yi;
            dz = z2-zi;

            xi = mod(xi,nx);
            yi = mod(yi,ny);
            zi = mod(zi,nz);

            % histogram indices
            ind = im2(1+xi+yi*nx+zi*nx*ny,:); 
            ind = 1 + [im1(:) ind(:)]; clear xi yi zi

            % function (m=1) and partial derivatives (m=2-7)
            cosdx = cospi(dx/2); sindx = sinpi(dx/2); clear dx
            cosdy = cospi(dy/2); sindy = sinpi(dy/2); clear dy
            cosdz = cospi(dz/2); sindz = sinpi(dz/2); clear dz

            for m = 1:7
                switch(m)
                    case 1; f = (1+cosdx).*(1+cosdy).*(1+cosdz);      % f
                    case 2; f = (0-sindx).*(1+cosdy).*(1+cosdz)*pi/2; % df/dx
                    case 3; f = (1+cosdx).*(0-sindy).*(1+cosdz)*pi/2; % df/dy
                    case 4; f = (1+cosdx).*(1+cosdy).*(0-sindz)*pi/2; % df/dz
                    case 5; f = sum(F.*(P*Dx),2)*pi/180;              % df/dxrot
                    case 6; f = sum(F.*(P*Dy),2)*pi/180;              % df/dyrot
                    case 7; f = sum(F.*(P*Dz),2)*pi/180;              % df/dzrot
                end

                % help with out-of-memory issues
                if m==2; F(:,1) = f; clear sindx; end
                if m==3; F(:,2) = f; clear sindy cosdz; end   
                if m==4; F(:,3) = f; clear sindz cosdx cosdy; end

                h(:,:,m) = h(:,:,m) + accumarray(ind,repmat(f,ns,1),[n n]);
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
fval = gather(HA+HB-HAB);

% partial derivatives
HA = dplogp(pA(:,2:7),pA(:,1));
HB = dplogp(pB(:,2:7),pB(:,1));
HAB = dplogp(h(:,2:7),h(:,1));
grad = gather(HA+HB-HAB);
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
