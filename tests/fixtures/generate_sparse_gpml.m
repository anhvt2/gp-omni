% Golden values for the gp-omni sparse GP (apxSparse) straight from gpml-matlab v4.2.
% Runs in Octave or MATLAB:
%     octave --no-gui tests/fixtures/generate_sparse_gpml.m
% Writes sparse_gpml_fixture.mat next to this script.

here = fileparts(mfilename('fullpath'));
gpml = getenv('GPML_DIR');
if isempty(gpml)
  gpml = '/Users/jingsun/Documents/gp/gp.GitHub/gpml-matlab/gpml-matlab-v4.2-2018-06-11';
end
addpath(gpml); startup;
rand('state', 1); randn('state', 1);

n = 40; D = 2; nu = 8;
x  = rand(n, D);
xs = rand(15, D);
yreg = sin(5*x(:,1)) + x(:,2).^2 + 0.1*randn(n,1);
ycls = sign(sin(4*x(:,1)) + x(:,2) - 0.5 + 0.2*randn(n,1)); ycls(ycls==0) = 1;
xu = x(1:nu, :) + 0.02*randn(nu, D);

out = struct();
out.x = x; out.xs = xs; out.yreg = yreg; out.ycls = ycls; out.xu = xu;

%% Gaussian likelihood, infGaussLik, FITC / SPEP / VFE, with hyp.xu and hyp.snu
cov = {@apxSparse, {@covSEard}, xu}; mean = @meanConst; lik = @likGauss;
hyp.mean = 0.1; hyp.cov = log([0.4; 0.6; 1.2]); hyp.lik = log(0.2); hyp.xu = xu;
ss = [1 0 0.5];
for i = 1:numel(ss)
  inf = @(varargin) infGaussLik(varargin{:}, struct('s', ss(i)));
  [nlZ, dnlZ] = gp(hyp, inf, mean, cov, lik, x, yreg);
  [ymu, ys2, fmu, fs2] = gp(hyp, inf, mean, cov, lik, x, yreg, xs);
  out.gauss(i) = struct('s', ss(i), 'nlZ', nlZ, 'dmean', dnlZ.mean, 'dcov', dnlZ.cov, ...
                        'dlik', dnlZ.lik, 'dxu', dnlZ.xu, 'ymu', ymu, 'ys2', ys2, ...
                        'fmu', fmu, 'fs2', fs2);
end
% learnable inducing noise hyp.snu (scalar), FITC
hs = hyp; hs.snu = log(3e-3);
inf = @(varargin) infGaussLik(varargin{:}, struct('s', 1));
[nlZ, dnlZ] = gp(hs, inf, mean, cov, lik, x, yreg);
out.snu = struct('nlZ', nlZ, 'dsnu', dnlZ.snu, 'dcov', dnlZ.cov);

%% Exact GP and Matern-5/2 + linear mean, for the nu == n limit and other covariances
hm = struct('mean', [0.3; -0.2; 0.05], 'cov', log([0.5; 0.7; 1.1]), 'lik', log(0.15));
meanlin = {@meanSum, {@meanLinear, @meanConst}};
[nlZe, dnlZe] = gp(hm, @infGaussLik, meanlin, {@covMaternard, 5}, @likGauss, x, yreg);
out.exact_matern = struct('nlZ', nlZe, 'dmean', dnlZe.mean, 'dcov', dnlZe.cov, 'dlik', dnlZe.lik);
hm2 = hm; hm2.xu = xu;
covm = {@apxSparse, {@covMaternard, 5}, xu};
inf = @(varargin) infGaussLik(varargin{:}, struct('s', 0));
[nlZm, dnlZm] = gp(hm2, inf, meanlin, covm, @likGauss, x, yreg);
out.sparse_matern = struct('nlZ', nlZm, 'dmean', dnlZm.mean, 'dcov', dnlZm.cov, ...
                           'dlik', dnlZm.lik, 'dxu', dnlZm.xu);

%% Laplace, probit (likErf) and logistic, sparse
hc.mean = 0.0; hc.cov = log([0.7; 0.9; 1.5]); hc.xu = xu;
liks = {@likErf, @likLogistic}; names = {'erf', 'logistic'};
k = 0;
for l = 1:2
  for i = 1:2
    s = ss(i); k = k + 1;
    opt = struct('s', s, 'irls_tol', 1e-13, 'irls_maxit', 200);
    inf = @(varargin) infLaplace(varargin{:}, opt);
    clear last_alpha;
    [nlZ, dnlZ] = gp(hc, inf, @meanConst, cov, liks{l}, x, ycls);
    [ymu, ys2, fmu, fs2] = gp(hc, inf, @meanConst, cov, liks{l}, x, ycls, xs);
    out.laplace(k) = struct('lik', names{l}, 's', s, 'nlZ', nlZ, 'dmean', dnlZ.mean, ...
                            'dcov', dnlZ.cov, 'dxu', dnlZ.xu, 'ymu', ymu, 'fmu', fmu, 'fs2', fs2);
  end
end

%% vfe_xu_opt: greedy inducing-point swap (VFE)
hv = struct('mean', 0.1, 'cov', log([0.4; 0.6; 1.2]), 'lik', log(0.2), 'xu', xu(1:5,:));
[hv2, nlZv] = vfe_xu_opt(hv, {@meanConst}, {@apxSparse, {@covSEard}, hv.xu}, x, yreg, x, 6);
out.xu_opt = struct('xu0', hv.xu, 'xu', hv2.xu, 'nlZ', nlZv);

save('-v6', fullfile(here, 'sparse_gpml_fixture.mat'), '-struct', 'out');
printf('wrote sparse_gpml_fixture.mat\n');
