% Generate ooDACE co-kriging reference values for gp-omni parity tests.
%
% Run in MATLAB (NOT Octave: ooDACE's classdef/import machinery does not load in Octave):
%     cd <repo>/tests/fixtures
%     addpath(genpath('<path to>/gp.GitHub/ooDACE')); startup;
%     generate_cokriging
% Writes cokriging_fixture.mat next to this script. Hyperparameters are fixed
% (no optimisation), so this checks only the model algebra, not the optimiser.
% NOTE: this script has not been executed by its author; check it runs before trusting it.

rng(0);
inDim = 2;
xl = rand(14, inDim); xh = rand(6, inDim);          % non-nested designs
hi = @(x) sin(5*x(:,1)) + x(:,2).^2;
lo = @(x) 0.8*hi(x) + 0.3*(x(:,1)-0.5);
samples = {xl, xh}; values = {lo(xl), hi(xh)};
xs = rand(25, inDim);

theta = [0.3 0.1];   % log10 precisions, shared by both levels (ooDACE uses hp0 for both)
rho   = 1.4;
cases = struct('corr', {}, 'regr', {}, 'mean', {}, 'var', {});
corrs = {@corrgauss, @corrmatern32};  corrNames = {'gauss', 'matern32'};
regrs = {'', 'regpoly0'};              regrNames = {'none', 'regpoly0'};
for c = 1:numel(corrs)
  for r = 1:numel(regrs)
    opts = CoKriging.getDefaultOptions();
    opts.type = 'CoKriging';
    opts.hpOptimizer = [];                 % use hp0 / rho0 as given
    opts.hp0 = theta;
    opts.hpBounds = [-2 -2; 2 2];
    opts.rho0 = rho;
    k = CoKriging(opts, theta, regrs{r}, corrs{c});
    k = k.fit(samples, values);
    [m, v] = k.predict(xs);
    cases(end+1) = struct('corr', corrNames{c}, 'regr', regrNames{r}, 'mean', m, 'var', v);
  end
end
save('-v6', 'cokriging_fixture.mat', 'samples', 'values', 'xs', 'theta', 'rho', 'cases');
fprintf('wrote cokriging_fixture.mat (%d cases)\n', numel(cases));
