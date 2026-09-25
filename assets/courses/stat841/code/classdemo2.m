function results = classdemo2(scenarioOrder, interactive, showFigures)
%CLASSDEMO Step-by-step classification and feature extraction lesson.
%
%   classdemo                     % All three datasets, with Next/Enter pauses
%   results = classdemo(2);        % Only dataset 2
%   results = classdemo(1:3,false,false);  % Calculations without figures
%
% MATLAB R2017b or newer. No data files or additional toolboxes are required.
% Press Next, Enter, or Space in the lesson window. Escape closes the lesson.
% There are 27 steps per dataset. Output contains completed datasets.
%
% COURSE REFERENCES (PDF page / printed slide number)
%   Lecture 2: pp. 16-27 / 72-83: labels 0/1, LDA and QDA delta_k(x).
%              p. 32 / 88: class priors, means and covariance estimates.
%   Lecture 3: pp. 16-19,27-32 / 121-124,132-137: centered PCA, S*u=lambda*u.
%   Lecture 4: pp. 6-13 / 143-150: FDA, S_B, S_W and w proportional to
%              S_W^(-1)*(mu_0-mu_1).
%              pp. 18-24 / 155-161: logistic likelihood and Newton-Raphson.
%
% CONVENTIONS
%   Displayed x is a column vector; stored XTrain/XTest have samples in rows.
%   Displayed PCA X contains centered observations in columns.
%   Displayed logistic X contains columns [1; x_i] to include an intercept.
%   To match slide 88, class covariance uses divisor n_k-d, where d is the
%   number of retained features. This is the slide's convention, not the
%   usual unbiased n_k-1 estimate. PCA S uses the usual sample divisor n-1.
%   Shared LDA covariance pools the corresponding within-class scatters.
%   FDA uses S_W=Sigma_0+Sigma_1 exactly as in Lecture 4, not pooled scatter.
%   PCA and FDA directions have unit Euclidean length for plotting.
%   This rescaling of FDA does not change its Rayleigh quotient.
%   Logistic regression maximizes the unpenalized likelihood in the slides.
%   Backtracking is a numerical safeguard; the displayed update includes
%   its step length if it is smaller than one.
%   All learned quantities use training observations only.
%   Each test sample is generated independently and used only for evaluation.

    if nargin < 1 || isempty(scenarioOrder), scenarioOrder = 1:3; end
    if nargin < 2 || isempty(interactive), interactive = true; end
    if nargin < 3 || isempty(showFigures), showFigures = true; end
    validateattributes(scenarioOrder,{'numeric'},{'vector','integer','>=',1,'<=',3});
    validateattributes(interactive,{'numeric','logical'},{'scalar'});
    validateattributes(showFigures,{'numeric','logical'},{'scalar'});
    showFigures = logical(showFigures);
    interactive = logical(interactive) && showFigures;
    scenarioOrder = scenarioOrder(:).';
    savedRng = rng;
    restoreRng = onCleanup(@() rng(savedRng)); %#ok<NASGU>
    results = struct([]);
    ui = [];
    if showFigures, ui = makeWindow(); end
    totalSteps = 27;
    stepNumber = 0;
    ctx = struct();

    for scenarioNumber = scenarioOrder
        D = makeData(scenarioNumber);
        X0 = D.XTrain(D.yTrain==0,:);
        X1 = D.XTrain(D.yTrain==1,:);
        n0 = size(X0,1); n1 = size(X1,1); n = n0+n1;
        d = size(D.XTrain,2);
        prior = [n0 n1]/n;
        mu = nan(2,d);
        Sigma = nan(d,d,2);
        ctx = struct('D',D,'mu',mu,'Sigma',Sigma,'prior',prior);
        stepNumber = 0;

        % 1-2: the observations and class labels.
        numbers = sprintf('Class 0: %d training points\nEach point: x = [x1; x2]',n0);
        if ~present('Class 0','Each blue dot is one observation.', ...
                {'$x=(x_1,x_2)^T$'},numbers,'class0'), return; end
        numbers = sprintf(['Class 0: n0 = %d\nClass 1: n1 = %d\nTotal:   n  = %d\n\n' ...
            'pi0 = %.4f\npi1 = %.4f\n\nBlue: class 0\nOrange: class 1'],n0,n1,n,prior);
        if ~present('Class labels and prior probabilities', ...
                'The prior probability is the fraction of training points in a class.', ...
                {'$\hat{\pi}_k=n_k/n,\qquad k\in\{0,1\}$'},numbers,'both'), return; end

        % 3-4: calculate the two means.
        mu(1,:) = mean(X0,1);
        ctx.mu = mu;
        numbers = sprintf('n0 = %d\nSum of x1 = %.4f\nSum of x2 = %.4f\n\nmu0 =\n%s', ...
            n0,sum(X0,1),matrixText(mu(1,:).'));
        if ~present('Mean of class 0','Average each coordinate of the blue points.', ...
                {'$\hat{\mu}_0=\frac{1}{n_0}\sum_{i:y_i=0}x_i$'},numbers,'mean0'), return; end
        mu(2,:) = mean(X1,1);
        ctx.mu = mu;
        numbers = sprintf('n1 = %d\nSum of x1 = %.4f\nSum of x2 = %.4f\n\nmu1 =\n%s', ...
            n1,sum(X1,1),matrixText(mu(2,:).'));
        if ~present('Mean of class 1','The crosses mark the two class means.', ...
                {'$\hat{\mu}_1=\frac{1}{n_1}\sum_{i:y_i=1}x_i$'},numbers,'means'), return; end

        % 5-6: use the covariance estimator shown in Lecture 2, slide 88.
        C0 = bsxfun(@minus,X0,mu(1,:));
        Sigma(:,:,1) = (C0.'*C0)/(n0-d);
        ctx.Sigma = Sigma;
        if ~present('Covariance of class 0', ...
                'The covariance describes how far the points spread in each direction.', ...
                {'$\hat{\Sigma}_0=\frac{\sum_{i:y_i=0}(x_i-\hat{\mu}_0)(x_i-\hat{\mu}_0)^T}{n_0-d}$'}, ...
                covarianceText(0,C0,Sigma(:,:,1),d),'cov0'), return; end
        C1 = bsxfun(@minus,X1,mu(2,:));
        Sigma(:,:,2) = (C1.'*C1)/(n1-d);
        ctx.Sigma = Sigma;
        if ~present('Covariance of class 1', ...
                'Each ellipse shows the spread described by its class covariance.', ...
                {'$\hat{\Sigma}_1=\frac{\sum_{i:y_i=1}(x_i-\hat{\mu}_1)(x_i-\hat{\mu}_1)^T}{n_1-d}$'}, ...
                covarianceText(1,C1,Sigma(:,:,2),d),'cov1'), return; end

        % 7-9: centered PCA, then one-dimensional coordinates.
        P = struct();
        P.mean = mean(D.XTrain,1);
        P.X = bsxfun(@minus,D.XTrain,P.mean).';
        P.S = P.X*P.X.'/(n-1);
        ctx.P = P;
        numbers = sprintf('Overall mean = [%.4f; %.4f]\n\nCentered X: 2 by %d\n\nS =\n%s', ...
            P.mean,n,matrixText(P.S));
        if ~present('PCA: center the data and calculate S', ...
                'Subtract the overall mean. Use all training points together.', ...
                {'$X=[x_1-\bar{x},\ldots,x_n-\bar{x}]$'; ...
                 '$S=\frac{1}{n-1}XX^T$'},numbers,'pcaCov'), return; end
        [V,E] = eig((P.S+P.S.')/2);
        [P.eigenvalues,order] = sort(diag(E),'descend');
        P.direction = V(:,order(1));
        [~,signIndex] = max(abs(P.direction));
        if P.direction(signIndex)<0, P.direction = -P.direction; end
        P.retainedVariance = P.eigenvalues(1)/sum(P.eigenvalues);
        P.train = (P.direction.'*P.X).';
        P.test = bsxfun(@minus,D.XTest,P.mean)*P.direction;
        ctx.P = P;
        numbers = sprintf(['Eigenvalues:\n  lambda1 = %.4f\n  lambda2 = %.4f\n\n' ...
            'u1 =\n%s\n\nVariance retained = %.2f%%'], ...
            P.eigenvalues,matrixText(P.direction),100*P.retainedVariance);
        if ~present('PCA: choose the first principal component', ...
                'Choose the direction with the largest variance. Class labels are not used.', ...
                {'$Su_1=\lambda_1u_1,\qquad u_1^Tu_1=1$'; ...
                 '$\lambda_1=\max_j\lambda_j$'},numbers,'pcaDirection'), return; end
        numbers = projectionExample('PCA',D.XTrain(1,:),P.mean,P.direction,P.train(1));
        if ~present('PCA: project from two dimensions to one', ...
                'Each point becomes one coordinate z. The star is the example calculated on the right.', ...
                {'$z=u_1^T(x-\bar{x}),\qquad Z=u_1^TX$'},numbers,'pcaProjection'), return; end

        % 10-13: FDA uses the between-class and within-class covariance.
        F = struct();
        F.delta = (mu(1,:)-mu(2,:)).';
        F.SB = F.delta*F.delta.';
        F.SW = Sigma(:,:,1)+Sigma(:,:,2);
        ctx.F = F;
        numbers = sprintf('mu0-mu1 =\n%s\n\nS_B =\n%s\n\nS_W =\n%s', ...
            matrixText(F.delta),matrixText(F.SB),matrixText(F.SW));
        if ~present('FDA: calculate S_B and S_W', ...
                'S_B measures the difference between the class means. S_W combines the class spreads.', ...
                {'$S_B=(\mu_0-\mu_1)(\mu_0-\mu_1)^T$'; ...
                 '$S_W=\Sigma_0+\Sigma_1$'},numbers,'fdaMatrices'), return; end
        F = fisherDirection(F);
        F.train = D.XTrain*F.direction;
        F.test = D.XTest*F.direction;
        P.J = fisherRatio(P.direction,F.SB,F.SW);
        ctx.F = F; ctx.P = P;
        numbers = sprintf('w =\n%s\n\nProjected mean, class 0 = %.4f\nProjected mean, class 1 = %.4f\n\nJ(w) = %.6f', ...
            matrixText(F.direction),mu*F.direction,F.J);
        explanation = 'Choose a direction that separates the class means relative to the class spreads.';
        if D.scenario==2
            explanation = 'The class means are nearly equal. FDA has little mean separation to use.';
        end
        if ~F.valid, explanation = 'The means are equal. There is no unique FDA direction.'; end
        if ~present('FDA: choose the projection direction',explanation, ...
                {'$J(w)=\frac{w^TS_Bw}{w^TS_Ww}$'; ...
                 '$w\propto S_W^{-1}(\mu_0-\mu_1)$'},numbers,'fdaDirection'), return; end
        numbers = projectionExample('FDA',D.XTrain(1,:),[0 0],F.direction,F.train(1));
        if ~present('FDA: project from two dimensions to one', ...
                'Project each point onto w. The star is the example calculated on the right.', ...
                {'$z=w^Tx=w_1x_1+w_2x_2$'},numbers,'fdaProjection'), return; end
        numbers = sprintf(['                    PCA        FDA\n' ...
            'Direction, x1    %9.4f  %9.4f\nDirection, x2    %9.4f  %9.4f\n' ...
            'Class 0 mean     %9.4f  %9.4f\nClass 1 mean     %9.4f  %9.4f\n' ...
            'Fisher ratio J   %9.4f  %9.4f'], ...
            P.direction(1),F.direction(1),P.direction(2),F.direction(2), ...
            mean(P.train(D.yTrain==0)),mean(F.train(D.yTrain==0)), ...
            mean(P.train(D.yTrain==1)),mean(F.train(D.yTrain==1)),P.J,F.J);
        if ~present('PCA and FDA: compare the one-dimensional features', ...
                'PCA keeps variance. FDA separates class means relative to class spreads.', ...
                {'$\mathrm{PCA}:\ \max_{u^Tu=1}u^TSu$'; ...
                 '$\mathrm{FDA}:\ \max_w\frac{w^TS_Bw}{w^TS_Ww}$'},numbers,'projectionCompare'), return; end

        % 14-16: shared covariance, LDA boundary, and a visible worked example.
        pooled = ((n0-d)*Sigma(:,:,1)+(n1-d)*Sigma(:,:,2))/(n-2*d);
        ctx.pooled = pooled;
        numbers = sprintf('Sigma0 =\n%s\n\nSigma1 =\n%s\n\nShared Sigma =\n%s', ...
            matrixText(Sigma(:,:,1)),matrixText(Sigma(:,:,2)),matrixText(pooled));
        if ~present('LDA: estimate the shared covariance', ...
                'LDA uses one covariance matrix for both classes.', ...
                {'$\Sigma=\frac{(n_0-d)\hat{\Sigma}_0+(n_1-d)\hat{\Sigma}_1}{n-2d}$'}, ...
                numbers,'pooled'), return; end
        lda = ldaFromMoments(mu,pooled,prior);
        ctx.lda = lda;
        numbers = sprintf(['a =\n%s\n\nb = %.5f\n\nDecision boundary:\n' ...
            '%.4f*x1 %+.4f*x2 %+.4f = 0'],matrixText(lda.linear),lda.constant, ...
            lda.linear(1),lda.linear(2),lda.constant);
        if ~present('LDA: linear decision boundary', ...
                'Choose the class with the larger discriminant value. The boundary is a straight line.', ...
                ldaFormula(),numbers,'lda'), return; end
        ctx.queryModel = lda;
        if ~present('LDA: calculate both discriminant values', ...
                'The star is x. Compare delta_0(x) and delta_1(x) to assign its class.', ...
                ldaFormula(),pointText(lda,D.query),'ldaPoint'), return; end

        % 17-19: QDA uses separate covariances and the slide's quadratic form.
        qda = qdaFromMoments(mu,Sigma,prior);
        ctx.qda = qda;
        numbers = sprintf('Sigma0 =\n%s\n\nSigma1 =\n%s\n\nlog|Sigma0| = %.5f\nlog|Sigma1| = %.5f', ...
            matrixText(Sigma(:,:,1)),matrixText(Sigma(:,:,2)), ...
            logDet(Sigma(:,:,1)),logDet(Sigma(:,:,2)));
        if ~present('QDA: use a separate covariance for each class', ...
                'QDA calculates delta_k(x) using the covariance of class k.', ...
                qdaFormula(),numbers,'qdaCov'), return; end
        explanation = 'The quadratic terms remain, so the decision boundary can curve.';
        if D.scenario==2
            explanation = 'QDA gives a closed curve here. The inside and outside belong to different classes.';
        end
        if ~present('QDA: quadratic decision boundary',explanation, ...
                {'$\delta_1(x)-\delta_0(x)=x^TAx+b^Tx+c$'; ...
                 '$\mathrm{Boundary}:\ x^TAx+b^Tx+c=0$'},quadraticText(qda),'qda'), return; end
        ctx.queryModel = qda;
        if ~present('QDA: calculate both discriminant values', ...
                'Calculate both values at the same star. The larger value determines the class.', ...
                qdaFormula(),pointText(qda,D.query),'qdaPoint'), return; end

        % 20-24: logistic likelihood and Newton-Raphson as in Lecture 4.
        XL = [ones(1,n);D.XTrain.'];
        y = D.yTrain;
        beta = zeros(d+1,1);
        trace = struct('beta',beta,'logLikelihood',logLikelihood(beta,XL,y),'stepSize',[]);
        ctx.logistic = logisticModel(beta); ctx.trace = trace;
        numbers = sprintf('beta = [0; 0; 0]\n\nP(Y=1|x) = 0.5 for every point\n\nLog-likelihood = %.4f',trace.logLikelihood);
        if ~present('Logistic regression: model the class probability', ...
                'The coefficients beta determine the probability of class 1.', ...
                logisticFormula(),numbers,'logisticInitial'), return; end
        for iteration = 1:2
            oldBeta = beta;
            [beta,detail] = newtonStep(beta,XL,y);
            trace = addIteration(trace,beta,detail);
            ctx.logistic = logisticModel(beta); ctx.trace = trace;
            numbers = sprintf(['beta before =\n%s\n\nX(y-p) =\n%s\n\n' ...
                'beta after =\n%s\n\nLog-likelihood: %.4f -> %.4f'], ...
                matrixText(oldBeta),matrixText(detail.gradient),matrixText(beta),detail.before,detail.after);
            formulas = {'$W_{ii}=p_i(1-p_i)$'; ...
                '$\beta^{t+1}=\beta^t+(XWX^T)^{-1}X(y-p)$'};
            if detail.alpha<1
                formulas{2} = '$\beta^{t+1}=\beta^t+\alpha(XWX^T)^{-1}X(y-p)$';
                numbers = [numbers sprintf('\nStep length alpha = %.4f',detail.alpha)];
            end
            if ~present(sprintf('Logistic regression: Newton update %d',iteration), ...
                    'Update beta. The log-likelihood increases as the probabilities fit the labels better.', ...
                    formulas,numbers,'logisticStep'), return; end
        end
        [beta,trace] = finishLogistic(beta,XL,y,trace);
        logistic = logisticModel(beta);
        ctx.logistic = logistic; ctx.trace = trace;
        numbers = sprintf(['beta =\n%s\n\nUpdates = %d\nLog-likelihood = %.4f\n\n' ...
            'Decision boundary:\n%.4f %+.4f*x1 %+.4f*x2 = 0'], ...
            matrixText(beta),size(trace.beta,2)-1,trace.logLikelihood(end),beta);
        if ~present('Logistic regression: fitted decision boundary', ...
                'The boundary is where the two class probabilities are equal.', ...
                {'$P(Y=1\mid x)=\frac{e^{\beta^T\tilde{x}}}{1+e^{\beta^T\tilde{x}}}$'; ...
                 '$P(Y=1\mid x)=\frac12\ \Longleftrightarrow\ \beta^T\tilde{x}=0$'}, ...
                numbers,'logisticFit'), return; end
        ctx.queryModel = logistic;
        numbers = sprintf(['x = [%.3f; %.3f]\n\nbeta^T*x_tilde = %.5f\n\n' ...
            'P(Y=1|x) = %.5f\nP(Y=0|x) = %.5f\n\nPredicted class = %d'], ...
            D.query,modelScore(logistic,D.query),sigmoid(modelScore(logistic,D.query)), ...
            1-sigmoid(modelScore(logistic,D.query)),predictClass(logistic,D.query));
        if ~present('Logistic regression: calculate the class probability', ...
                'Assign class 1 when its probability is greater than 0.5.', ...
                logisticFormula(),numbers,'logisticPoint'), return; end

        % 25: evaluate the original two-dimensional models.
        models = {lda,qda,logistic};
        [trainAcc,testAcc,confusion] = evaluateModels(models,D.XTrain,D.XTest,D.yTrain,D.yTest);
        ctx.models = models; ctx.trainAcc = trainAcc; ctx.testAcc = testAcc;
        if ~present('Classification with the original two features', ...
                'Dot color is the true class. Black crosses mark test errors.', ...
                {},evaluationText(trainAcc,testAcc,confusion,numel(D.yTest)),'comparison'), return; end

        % 26: use PCA/FDA as preprocessing for each of the same classifiers.
        projectionModels = cell(2,3);
        projectionAccuracy = nan(2,3);
        projectedTrainAccuracy = nan(2,3);
        projections = {P,F};
        for r = 1:2
            if r==2 && ~F.valid, continue; end
            Z = projections{r};
            projectionModels(r,:) = fitModels(Z.train,D.yTrain);
            [projectedTrainAccuracy(r,:),projectionAccuracy(r,:)] = ...
                evaluateModels(projectionModels(r,:),Z.train,Z.test,D.yTrain,D.yTest);
        end
        accuracyByFeatures = [testAcc;projectionAccuracy];
        ctx.accuracyByFeatures = accuracyByFeatures;
        numbers = sprintf(['Test accuracy (%%)\n\nFeatures        LDA       QDA    Logistic\n' ...
            'Original 2D  %8.2f  %8.2f  %8.2f\n' ...
            'PCA 1D       %8.2f  %8.2f  %8.2f\n' ...
            'FDA 1D       %8.2f  %8.2f  %8.2f\n\n' ...
            'Same test observations in all nine comparisons.'],100*accuracyByFeatures.');
        if ~present('PCA and FDA as preprocessing: compare test accuracy', ...
                'Fit each classifier again using one PCA feature or one FDA feature.', ...
                {'$\mathrm{PCA}:\ x\mapsto u_1^T(x-\bar{x})$'; ...
                 '$\mathrm{FDA}:\ x\mapsto w^Tx$'},numbers,'featureAccuracy'), return; end

        % 27: known generating distributions explain the fitted boundaries.
        bayesAccuracy = mean(double(trueScore(D,D.XTest)>0)==D.yTest);
        numbers = sprintf('%s\n\nBayes test accuracy = %.2f%%\n\nSolid: fitted boundary\nDashed: Bayes boundary', ...
            D.truthText,100*bayesAccuracy);
        if ~present('Compare with the Bayes decision boundary',D.takeaway, ...
                D.truthFormula,numbers,'truth'), return; end

        result = struct('scenario',scenarioNumber,'name',D.name,'data',D, ...
            'mu',mu,'Sigma',Sigma,'pooledSigma',pooled,'prior',prior, ...
            'PCA',P,'FDA',F,'models',{models},'logisticTrace',trace, ...
            'trainingAccuracy',trainAcc,'testAccuracy',testAcc,'confusion',confusion, ...
            'projectionModels',{projectionModels},'accuracyByFeatures',accuracyByFeatures, ...
            'projectedTrainAccuracy',projectedTrainAccuracy, ...
            'bayesAccuracy',bayesAccuracy,'stepsCompleted',stepNumber);
        results(end+1) = result; %#ok<AGROW>
    end
    if showFigures && isgraphics(ui.fig), showSummary(ui,results); end

    function keepGoing = present(titleText,explanation,formulas,numbers,kind)
        stepNumber = stepNumber+1;
        fprintf('\nDataset %d | Step %d/%d | %s\n%s\n\n%s\n', ...
            scenarioNumber,stepNumber,totalSteps,titleText,explanation,numbers);
        keepGoing = true;
        if ~showFigures, return; end
        if ~isgraphics(ui.fig), keepGoing = false; return; end
        set(ui.heading,'String',titleText);
        set(ui.stepTitle,'String',sprintf('Dataset %d: %s   |   Step %02d / %d', ...
            scenarioNumber,D.name,stepNumber,totalSteps));
        layout = 'standard';
        if any(strcmp(kind,{'comparison','featureAccuracy','truth'})), layout = 'wide'; end
        if strcmp(kind,'projectionCompare'), layout = 'projections'; end
        arrangeWindow(ui,layout,isempty(formulas));
        set(ui.explanation,'String',wrapWords(explanation,56));
        set(ui.numbers,'String',numbers);
        showFormulas(ui.formulaAxes,formulas);
        setappdata(ui.fig,'AdvanceRequested',false);
        setappdata(ui.fig,'Waiting',interactive);
        renderStep(ui.plotPanel,kind,ctx);
        drawnow;
        if ~isgraphics(ui.fig), keepGoing = false; return; end
        if interactive && ~getappdata(ui.fig,'AdvanceRequested'), uiwait(ui.fig); end
        keepGoing = isgraphics(ui.fig);
        if keepGoing, setappdata(ui.fig,'Waiting',false); end
    end
end

%% Data generation: separate random draws for training and testing.
function D = makeData(scenario)
    rng(100+scenario,'twister');
    D.scenario = scenario;
    if scenario==1
        D.name = 'Two tall groups';
        D.trueMu = [-1.5 0;1.5 0];
        D.trueSigma = repmat(diag([0.64 9]),[1 1 2]);
        D.query = [0.4 0.6];
        D.truthText = sprintf('mu0 = [-1.5; 0]\nmu1 = [ 1.5; 0]\nSigma0 = Sigma1 = diag(0.64,9)');
        D.truthFormula = {'$\Sigma_0=\Sigma_1,\quad\pi_0=\pi_1$';'$\mathrm{Bayes\ boundary}:\ x_1=0$'};
        D.takeaway = 'The true boundary is a straight line. FDA keeps the horizontal class separation; PCA mainly keeps vertical spread.';
    elseif scenario==2
        D.name = 'Same center, different spread';
        D.trueMu = zeros(2,2);
        D.trueSigma = cat(3,0.25*eye(2),4*eye(2));
        D.query = [0.3 0.2];
        D.truthText = sprintf(['mu0 = mu1 = [0; 0]\nSigma0 = 0.25*I; Sigma1 = 4*I\n\n' ...
            'x1^2+x2^2 = (8/15)*log(16)\nRadius = %.4f'],sqrt((8/15)*log(16)));
        D.truthFormula = {'$\delta_1-\delta_0=\frac{15}{8}(x_1^2+x_2^2)-\log16$'; ...
            '$\mathrm{Boundary}:\ x_1^2+x_2^2=\frac{8}{15}\log16$'};
        D.takeaway = 'The true boundary is a circle. QDA can represent it. The two class means are the same.';
    else
        D.name = 'Three groups of points';
        D.query = [0.8 1.5];
        D.truthText = sprintf('X is drawn from three Gaussian groups.\n\nP(Y=1|x) = exp(2*x1)/(1+exp(2*x1))');
        D.truthFormula = {'$P(Y=1\mid x)=\frac{e^{2x_1}}{1+e^{2x_1}}$'; ...
            '$\mathrm{Bayes\ boundary}:\ x_1=0$'};
        D.takeaway = 'Here the true probability follows a logistic model. The class distributions are not Gaussian.';
        [D.XTrain,D.yTrain] = mixtureSample(600);
        [D.XTest,D.yTest] = mixtureSample(2400);
    end
    if scenario<=2
        [D.XTrain,D.yTrain] = gaussianSample(160,D.trueMu,D.trueSigma);
        [D.XTest,D.yTest] = gaussianSample(600,D.trueMu,D.trueSigma);
    end
    lo = min(D.XTrain,[],1); hi = max(D.XTrain,[],1);
    padding = 0.12*max(hi-lo,1);
    D.fullLimits = [lo(1)-padding(1),hi(1)+padding(1),lo(2)-padding(2),hi(2)+padding(2)];
    D.boundaryLimits = D.fullLimits;
    if scenario==3, D.boundaryLimits = [-3.5 3.5 -3.5 3.5]; end
end

function [X,y] = gaussianSample(n,mu,Sigma)
    X = [bsxfun(@plus,randn(n,2)*chol(Sigma(:,:,1)),mu(1,:)); ...
         bsxfun(@plus,randn(n,2)*chol(Sigma(:,:,2)),mu(2,:))];
    y = [zeros(n,1);ones(n,1)];
end

function [X,y] = mixtureSample(n)
    u = rand(n,1);
    group = zeros(n,1);
    group(u<0.08) = -1; group(u>0.92) = 1;
    X = randn(n,2)+bsxfun(@times,group,[16 8]);
    y = double(rand(n,1)<sigmoid(2*X(:,1)));
end

%% PCA and Fisher quantities.
function F = fisherDirection(F)
    F.valid = norm(F.delta)>1e-12;
    if F.valid
        direction = F.SW\F.delta;
        F.direction = direction/norm(direction);
        F.J = fisherRatio(F.direction,F.SB,F.SW);
    else
        % Do not invent a separating direction when S_B is zero.
        F.direction = nan(size(F.delta));
        F.J = 0;
    end
end

function J = fisherRatio(direction,SB,SW)
    J = (direction.'*SB*direction)/(direction.'*SW*direction);
end

%% Models support either the original two features or one projected feature.
function [mu,Sigma,pooled,prior] = moments(X,y)
    d = size(X,2); n = size(X,1);
    mu = zeros(2,d); Sigma = zeros(d,d,2);
    counts = zeros(1,2); scatter = zeros(d);
    for k = 0:1
        Xi = X(y==k,:); counts(k+1) = size(Xi,1);
        mu(k+1,:) = mean(Xi,1);
        centered = bsxfun(@minus,Xi,mu(k+1,:));
        Si = centered.'*centered;
        Sigma(:,:,k+1) = Si/(counts(k+1)-d);
        scatter = scatter+Si;
    end
    pooled = scatter/(n-2*d);
    prior = counts/n;
end

function model = ldaFromMoments(mu,Sigma,prior)
    d = size(mu,2);
    model = struct('name','LDA','kind','lda','A',zeros(d), ...
        'linear',Sigma\(mu(2,:)-mu(1,:)).','constant',0, ...
        'mu',mu,'Sigma',Sigma,'prior',prior);
    model.constant = -0.5*(mu(2,:)*(Sigma\mu(2,:).') ...
        -mu(1,:)*(Sigma\mu(1,:).'))+log(prior(2)/prior(1));
end

function model = qdaFromMoments(mu,Sigma,prior)
    d = size(mu,2);
    P0 = Sigma(:,:,1)\eye(d); P1 = Sigma(:,:,2)\eye(d);
    model = struct('name','QDA','kind','qda','A',0.5*(P0-P1), ...
        'linear',P1*mu(2,:).'-P0*mu(1,:).','constant',0, ...
        'mu',mu,'Sigma',Sigma,'prior',prior);
    model.constant = -0.5*(mu(2,:)*P1*mu(2,:).'-mu(1,:)*P0*mu(1,:).') ...
        -0.5*(logDet(Sigma(:,:,2))-logDet(Sigma(:,:,1)))+log(prior(2)/prior(1));
end

function value = logDet(Sigma)
    value = 2*sum(log(diag(chol(Sigma))));
end

function model = logisticModel(beta)
    model = struct('name','Logistic','kind','logistic','A',zeros(numel(beta)-1), ...
        'linear',beta(2:end),'constant',beta(1),'beta',beta);
end

function value = modelScore(model,X)
    value = sum((X*model.A).*X,2)+X*model.linear+model.constant;
end

function y = predictClass(model,X)
    y = double(modelScore(model,X)>0);
end

function [values,terms] = discriminants(model,X)
    n = size(X,1);
    values = zeros(n,2); terms = zeros(n,3,2);
    for k = 1:2
        if strcmp(model.kind,'lda')
            terms(:,1,k) = X*(model.Sigma\model.mu(k,:).');
            terms(:,2,k) = -0.5*model.mu(k,:)*(model.Sigma\model.mu(k,:).');
        else
            centered = bsxfun(@minus,X,model.mu(k,:));
            terms(:,1,k) = -0.5*logDet(model.Sigma(:,:,k));
            terms(:,2,k) = -0.5*sum((centered/model.Sigma(:,:,k)).*centered,2);
        end
        terms(:,3,k) = log(model.prior(k));
        values(:,k) = sum(terms(:,:,k),2);
    end
end

function p = sigmoid(z)
    p = zeros(size(z)); positive = z>=0;
    p(positive) = 1./(1+exp(-z(positive)));
    t = exp(z(~positive));
    p(~positive) = t./(1+t);
end

function ell = logLikelihood(beta,X,y)
    eta = X.'*beta;
    ell = sum(y.*eta-max(eta,0)-log1p(exp(-abs(eta))));
end

function [updated,detail] = newtonStep(beta,X,y)
    p = sigmoid(X.'*beta);
    gradient = X*(y-p);
    info = bsxfun(@times,X,(p.*(1-p)).')*X.';
    if rcond(info)<1e-12
        direction = pinv(info)*gradient;
    else
        direction = info\gradient;
    end
    before = logLikelihood(beta,X,y);
    alpha = 1;
    while alpha>=2^-24
        updated = beta+alpha*direction;
        after = logLikelihood(updated,X,y);
        if after>=before+1e-4*alpha*(gradient.'*direction)-1e-10
            break;
        end
        alpha = alpha/2;
    end
    if alpha<2^-24, error('classdemo:NewtonStep','No increasing likelihood step was found.'); end
    detail = struct('gradient',gradient,'information',info,'alpha',alpha, ...
        'before',before,'after',after);
end

function trace = addIteration(trace,beta,detail)
    trace.beta(:,end+1) = beta;
    trace.logLikelihood(end+1) = detail.after;
    trace.stepSize(end+1) = detail.alpha;
end

function [beta,trace] = finishLogistic(beta,X,y,trace)
    converged = false;
    for iteration = size(trace.beta,2):80
        [next,detail] = newtonStep(beta,X,y);
        trace = addIteration(trace,next,detail);
        small = norm(next-beta)<=1e-8*(1+norm(beta));
        beta = next;
        if small || norm(detail.gradient,inf)<1e-8
            converged = true; break;
        end
    end
    trace.converged = converged;
    if ~converged
        warning('classdemo:Convergence','Logistic coefficients did not converge in 80 updates.');
    end
end

function models = fitModels(X,y)
    [mu,Sigma,pooled,prior] = moments(X,y);
    XL = [ones(1,size(X,1));X.'];
    beta = zeros(size(XL,1),1);
    trace = struct('beta',beta,'logLikelihood',logLikelihood(beta,XL,y),'stepSize',[]);
    [beta,~] = finishLogistic(beta,XL,y,trace);
    models = {ldaFromMoments(mu,pooled,prior),qdaFromMoments(mu,Sigma,prior),logisticModel(beta)};
end

function [trainAcc,testAcc,C] = evaluateModels(models,XTrain,XTest,yTrain,yTest)
    trainAcc = zeros(1,3); testAcc = zeros(1,3); C = zeros(2,2,3);
    for j = 1:3
        trainPrediction = predictClass(models{j},XTrain);
        testPrediction = predictClass(models{j},XTest);
        trainAcc(j) = mean(trainPrediction==yTrain);
        testAcc(j) = mean(testPrediction==yTest);
        for truth = 0:1
            for prediction = 0:1
                C(truth+1,prediction+1,j) = sum(yTest==truth & testPrediction==prediction);
            end
        end
    end
end

function value = trueScore(D,X)
    if D.scenario<=2
        model = qdaFromMoments(D.trueMu,D.trueSigma,[.5 .5]);
        value = modelScore(model,X);
    else
        value = 2*X(:,1);
    end
end

%% Formulas and worked numerical examples.
function formulas = ldaFormula()
    formulas = {'$\delta_k(x)=x^T\Sigma^{-1}\mu_k-\frac12\mu_k^T\Sigma^{-1}\mu_k+\log\pi_k$'; ...
        '$\mathrm{Boundary}:\ \delta_1(x)=\delta_0(x)\ \Longleftrightarrow\ a^Tx+b=0$'};
end

function formulas = qdaFormula()
    formulas = {'$\delta_k(x)=-\frac12\log|\Sigma_k|$'; ...
        '$\quad-\frac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+\log\pi_k$'; ...
        '$h(x)=\arg\max_{k\in\{0,1\}}\delta_k(x)$'};
end

function formulas = logisticFormula()
    formulas = {'$P(Y=1\mid x)=\frac{e^{\beta^T\tilde{x}}}{1+e^{\beta^T\tilde{x}}}$'; ...
        '$\tilde{x}=(1,x_1,x_2)^T$'};
end

function str = matrixText(M)
    rows = cell(size(M,1),1);
    format = ['[' repmat('% .4f  ',1,size(M,2)) ']'];
    for k = 1:size(M,1), rows{k} = sprintf(format,M(k,:)); end
    str = strjoin(rows,sprintf('\n'));
end

function str = covarianceText(k,C,Sigma,d)
    str = sprintf(['First centered point:\nx-mu%d = [%.3f; %.3f]\n\n' ...
        'Its matrix contribution:\n%s\n\n' ...
        'Number of points = %d\nDivisor n%d-d = %d\n\nSigma%d =\n%s'], ...
        k,C(1,:),matrixText(C(1,:).'*C(1,:)),size(C,1),k,size(C,1)-d,k,matrixText(Sigma));
end

function str = projectionExample(method,x,center,v,z)
    if any(~isfinite(v))
        str = sprintf('The sample means are equal.\nNo unique FDA direction is available.');
        return;
    end
    input = x-center;
    if strcmp(method,'PCA')
        first = sprintf('x = [%.3f; %.3f]\nx-mean = [%.3f; %.3f]',x,input);
    else
        first = sprintf('x = [%.3f; %.3f]',x);
    end
    str = sprintf(['%s\n\nDirection = [%.4f; %.4f]\n\n' ...
        'z = (%.4f)*(%.4f)\n  + (%.4f)*(%.4f)\n  = %.4f\n\n2 coordinates -> 1 coordinate'], ...
        first,v,v(1),input(1),v(2),input(2),z);
end

function str = pointText(model,x)
    [delta,terms] = discriminants(model,x);
    str = sprintf(['x = [%.3f; %.3f]\n\n' ...
        'delta_0(x):\n  %.4f %+.4f %+.4f\n  = %.5f\n\n' ...
        'delta_1(x):\n  %.4f %+.4f %+.4f\n  = %.5f\n\n' ...
        'delta_1 - delta_0 = %.5f\n\nPredicted class = %d'], ...
        x,terms(:,:,1),delta(1),terms(:,:,2),delta(2),delta(2)-delta(1),predictClass(model,x));
end

function str = quadraticText(model)
    str = sprintf(['A =\n%s\n\nb =\n%s\n\nc = %.5f\n\n' ...
        'Boundary:\n%.3f*x1^2 %+.3f*x1*x2\n' ...
        '%+.3f*x2^2 %+.3f*x1\n%+.3f*x2 %+.3f = 0'], ...
        matrixText(model.A),matrixText(model.linear),model.constant, ...
        model.A(1,1),model.A(1,2)+model.A(2,1),model.A(2,2),model.linear,model.constant);
end

function str = evaluationText(trainAcc,testAcc,C,nTest)
    str = sprintf(['              LDA       QDA    Logistic\n' ...
        'Train (%%)   %7.2f   %7.2f   %7.2f\nTest  (%%)   %7.2f   %7.2f   %7.2f\n\n' ...
        'Confusion: rows=true, columns=predicted\n' ...
        'Class 0     %4d %4d  %4d %4d  %4d %4d\n' ...
        'Class 1     %4d %4d  %4d %4d  %4d %4d\n\n' ...
        'All %d test points are evaluated; up to 360 are drawn.'], ...
        100*trainAcc,100*testAcc, ...
        C(1,1,1),C(1,2,1),C(1,1,2),C(1,2,2),C(1,1,3),C(1,2,3), ...
        C(2,1,1),C(2,2,1),C(2,1,2),C(2,2,2),C(2,1,3),C(2,2,3),nTest);
end

%% One lesson window; Enter/Next releases one waiting step.
function ui = makeWindow()
    ui.fig = figure('Name','classdemo','NumberTitle','off','Color','w', ...
        'MenuBar','none','ToolBar','none','WindowStyle','normal', ...
        'Units','normalized','Position',[.025 .055 .95 .875], ...
        'WindowKeyPressFcn',@lessonKey,'CloseRequestFcn',@(f,e) delete(f));
    setappdata(ui.fig,'Waiting',false);
    setappdata(ui.fig,'AdvanceRequested',false);
    ui.heading = uicontrol(ui.fig,'Style','text','Units','normalized', ...
        'Position',[.035 .927 .93 .054],'BackgroundColor','w', ...
        'HorizontalAlignment','left','FontSize',20,'FontWeight','bold');
    ui.stepTitle = uicontrol(ui.fig,'Style','text','Units','normalized', ...
        'Position',[.035 .88 .93 .042],'BackgroundColor','w', ...
        'HorizontalAlignment','left','FontSize',13,'ForegroundColor',[.25 .3 .37]);
    ui.plotPanel = uipanel(ui.fig,'Units','normalized','BorderType','none','BackgroundColor','w');
    ui.notePanel = uipanel(ui.fig,'Units','normalized','BorderType','none','BackgroundColor','w');
    ui.explanation = uicontrol(ui.notePanel,'Style','text','Units','normalized', ...
        'BackgroundColor','w','HorizontalAlignment','left','FontSize',12);
    ui.formulaAxes = axes('Parent',ui.notePanel,'Units','normalized','Visible','off');
    ui.numbers = uicontrol(ui.notePanel,'Style','edit','Min',0,'Max',2, ...
        'Enable','inactive','Units','normalized','HorizontalAlignment','left', ...
        'BackgroundColor',[.95 .97 .99],'FontSize',11.5,'FontName',get(0,'FixedWidthFontName'));
    ui.next = uicontrol(ui.fig,'Style','pushbutton','Units','normalized', ...
        'Position',[.79 .022 .175 .055],'String','Next / Enter','FontSize',13, ...
        'FontWeight','bold','Callback',@(s,e) requestAdvance(ancestor(s,'figure')), ...
        'KeyPressFcn',@(s,e) lessonKey(ancestor(s,'figure'),e));
    uicontrol(ui.fig,'Style','pushbutton','Units','normalized','Position',[.035 .022 .11 .055], ...
        'String','Close','FontSize',11,'Callback',@(s,e) delete(ancestor(s,'figure')));
    uicontrol(ui.fig,'Style','text','Units','normalized','Position',[.17 .025 .59 .043], ...
        'String','Enter / Space: next step     Esc: close', ...
        'BackgroundColor','w','FontSize',11,'HorizontalAlignment','left');
end

function requestAdvance(fig)
    if isgraphics(fig) && getappdata(fig,'Waiting')
        setappdata(fig,'AdvanceRequested',true);
        uiresume(fig);
    end
end

function lessonKey(fig,event)
    if any(strcmp(event.Key,{'return','enter','space'}))
        requestAdvance(fig);
    elseif strcmp(event.Key,'escape') && isgraphics(fig)
        delete(fig);
    end
end

function arrangeWindow(ui,layout,noFormulas)
    if strcmp(layout,'projections')
        set(ui.plotPanel,'Position',[.025 .30 .95 .56]);
        set(ui.notePanel,'Position',[.035 .095 .93 .18]);
        set(ui.explanation,'Position',[0 .64 .40 .35]);
        set(ui.formulaAxes,'Position',[.01 .01 .39 .61]);
        set(ui.numbers,'Position',[.43 .01 .57 .98],'FontSize',10.5);
    elseif strcmp(layout,'wide')
        set(ui.plotPanel,'Position',[.025 .405 .95 .455]);
        set(ui.notePanel,'Position',[.035 .095 .93 .29]);
        set(ui.explanation,'Position',[0 .63 .40 .35]);
        set(ui.formulaAxes,'Position',[.01 .03 .39 .53]);
        set(ui.numbers,'Position',[.43 .01 .57 .98],'FontSize',10.5);
    else
        set(ui.plotPanel,'Position',[.02 .12 .555 .73]);
        set(ui.notePanel,'Position',[.60 .11 .365 .75]);
        set(ui.explanation,'Position',[0 .88 1 .115]);
        set(ui.formulaAxes,'Position',[.01 .665 .98 .20]);
        set(ui.numbers,'Position',[0 0 1 .645],'FontSize',11.5);
    end
    if noFormulas
        set(ui.explanation,'Position',[0 .68 .40 .30]);
    end
end

function showFormulas(ax,formulas)
    cla(ax);
    set(ax,'Visible','off','XLim',[0 1],'YLim',[0 1]);
    if isempty(formulas), return; end
    h = text(ax,.01,.96,formulas,'Units','normalized','Interpreter','latex', ...
        'VerticalAlignment','top','FontSize',15,'Color',[.12 .22 .40]);
    % Fit the actual rendered formula, not its source-code character count.
    extent = get(h,'Extent');
    while extent(3)>.97 && get(h,'FontSize')>10
        set(h,'FontSize',get(h,'FontSize')-.5);
        extent = get(h,'Extent');
    end
end

function lines = wrapWords(str,width)
    words = strsplit(str); lines = {}; current = '';
    for k = 1:numel(words)
        if isempty(current)
            current = words{k};
        elseif numel(current)+numel(words{k})+1<=width
            current = [current ' ' words{k}]; %#ok<AGROW>
        else
            lines{end+1} = current; current = words{k}; %#ok<AGROW>
        end
    end
    if ~isempty(current), lines{end+1} = current; end
end

%% Plot each teaching step.
function renderStep(panel,kind,C)
    delete(get(panel,'Children'));
    D = C.D;
    if strcmp(kind,'projectionCompare')
        edges = sharedEdges([C.P.train;C.F.train],30);
        directions = {C.P,C.F};
        methods = {'PCA','FDA'};
        for j = 1:2
            left = .065+(j-1)*.49;
            ax = axes('Parent',panel,'Position',[left .50 .39 .45]);
            drawProjectionDirection(ax,D,directions{j},methods{j},false);
            title(ax,[methods{j} ' direction'],'FontSize',12);
            ax1 = axes('Parent',panel,'Position',[left .11 .39 .22]);
            drawOneDimension(ax1,directions{j}.train,D.yTrain,edges);
            title(ax1,[methods{j} ': one feature'],'FontSize',12);
        end
        return;
    end
    if any(strcmp(kind,{'pcaProjection','fdaProjection'}))
        if strcmp(kind,'pcaProjection'), F = C.P; method = 'PCA';
        else, F = C.F; method = 'FDA'; end
        ax = axes('Parent',panel,'Position',[.12 .52 .82 .42]);
        drawProjectionDirection(ax,D,F,method,true);
        title(ax,[method ': projection onto a line']);
        ax1 = axes('Parent',panel,'Position',[.12 .12 .82 .24]);
        drawOneDimension(ax1,F.train,D.yTrain,sharedEdges(F.train,28));
        title(ax1,[method ': one-dimensional class distributions']);
        return;
    end
    if strcmp(kind,'featureAccuracy')
        ax = axes('Parent',panel,'Position',[.075 .19 .87 .73]);
        drawAccuracyBars(ax,C.accuracyByFeatures);
        title(ax,'Test accuracy using the original features or one extracted feature');
        return;
    end
    if any(strcmp(kind,{'comparison','truth'}))
        for j = 1:3
            ax = axes('Parent',panel,'Position',[.045+(j-1)*.332 .17 .27 .71]);
            drawDecision(ax,C.models{j},D.boundaryLimits);
            if strcmp(kind,'comparison')
                idx = round(linspace(1,numel(D.yTest),min(360,numel(D.yTest))));
                drawClasses(ax,D.XTest(idx,:),D.yTest(idx),0:1,12);
                prediction = predictClass(C.models{j},D.XTest(idx,:));
                errors = idx(prediction~=D.yTest(idx));
                plot(ax,D.XTest(errors,1),D.XTest(errors,2),'kx','MarkerSize',5,'HandleVisibility','off');
                title(ax,sprintf('%s: %.1f%%',C.models{j}.name,100*C.testAcc(j)),'FontSize',12);
            else
                drawClasses(ax,D.XTrain,D.yTrain,0:1,10);
                [g1,g2,points] = makeGrid(D.boundaryLimits);
                addBoundary(ax,g1,g2,reshape(trueScore(D,points),size(g1)),'k--',2.3);
                title(ax,[C.models{j}.name ' and Bayes boundary'],'FontSize',11);
            end
            finishAxes(ax,D.boundaryLimits);
            if D.scenario==3, xlabel(ax,'x_1 (central region)'); end
        end
        return;
    end
    if strcmp(kind,'logisticFit')
        ax = axes('Parent',panel,'Position',[.12 .49 .82 .43]);
        drawDecision(ax,C.logistic,D.boundaryLimits);
        drawClasses(ax,D.XTrain,D.yTrain,0:1,14);
        finishAxes(ax,D.boundaryLimits); title(ax,'Logistic decision boundary');
        ax1 = axes('Parent',panel,'Position',[.12 .12 .82 .22]);
        plot(ax1,0:numel(C.trace.logLikelihood)-1,C.trace.logLikelihood,'o-', ...
            'Color',[.12 .36 .65],'LineWidth',1.8);
        xlabel(ax1,'Newton update'); ylabel(ax1,'Log-likelihood');
        grid(ax1,'on'); box(ax1,'on');
        return;
    end

    ax = axes('Parent',panel,'Position',[.12 .13 .82 .78]);
    hold(ax,'on');
    if strcmp(kind,'pcaDirection')
        drawProjectionDirection(ax,D,C.P,'PCA',false);
        title(ax,'First principal component u_1');
        return;
    elseif strcmp(kind,'fdaDirection')
        drawProjectionDirection(ax,D,C.F,'FDA',false);
        title(ax,'Fisher direction w');
        return;
    elseif strcmp(kind,'pcaCov')
        centered = C.P.X.';
        scatter(ax,centered(:,1),centered(:,2),18,[.4 .45 .52],'filled');
        plot(ax,0,0,'kx','LineWidth',2,'MarkerSize',14);
        limits = D.fullLimits-[C.P.mean(1) C.P.mean(1) C.P.mean(2) C.P.mean(2)];
        finishAxes(ax,limits);
        xlabel(ax,'x_1 - mean(x_1)'); ylabel(ax,'x_2 - mean(x_2)');
        title(ax,'Centered training data');
        return;
    end

    limits = D.fullLimits;
    decisionKinds = {'lda','ldaPoint','qda','qdaPoint','logisticInitial','logisticStep','logisticPoint'};
    decision = any(strcmp(kind,decisionKinds));
    if decision
        limits = D.boundaryLimits;
        if any(strcmp(kind,{'lda','ldaPoint'})), model = C.lda;
        elseif any(strcmp(kind,{'qda','qdaPoint'})), model = C.qda;
        else, model = C.logistic; end
        if ~strcmp(kind,'logisticInitial'), drawDecision(ax,model,limits); end
    end
    classes = 0:1;
    if strcmp(kind,'class0'), classes = 0; end
    drawClasses(ax,D.XTrain,D.yTrain,classes,19);
    colors = classColors();
    if ~decision
        for k = 1:2
            if all(isfinite(C.mu(k,:)))
                plot(ax,C.mu(k,1),C.mu(k,2),'x','Color',colors(k,:), ...
                    'LineWidth',3,'MarkerSize',15,'HandleVisibility','off');
                text(ax,C.mu(k,1),C.mu(k,2),sprintf('  mu_%d',k-1), ...
                    'Color',colors(k,:),'FontWeight','bold','FontSize',12);
            end
        end
        if any(strcmp(kind,{'cov0','cov1','pooled','qdaCov','fdaMatrices'}))
            for k = 1:2
                S = C.Sigma(:,:,k);
                if strcmp(kind,'pooled'), S = C.pooled; end
                if all(isfinite(S(:))), drawEllipse(ax,C.mu(k,:),S,colors(k,:)); end
            end
        end
        if any(strcmp(kind,{'cov0','cov1'}))
            k = double(strcmp(kind,'cov1'));
            idx = find(D.yTrain==k,1);
            x = D.XTrain(idx,:);
            plot(ax,[C.mu(k+1,1),x(1)],[C.mu(k+1,2),x(2)],'k--', ...
                'LineWidth',1.3,'HandleVisibility','off');
            plot(ax,x(1),x(2),'ko','MarkerSize',10,'LineWidth',1.3, ...
                'DisplayName','Point used in the calculation');
        end
    end
    if any(strcmp(kind,{'ldaPoint','qdaPoint','logisticPoint'}))
        plot(ax,D.query(1),D.query(2),'kp','MarkerSize',18, ...
            'MarkerFaceColor',[1 .9 .2],'DisplayName','x');
        showPointResult(ax,model,D.query);
    end
    finishAxes(ax,limits);
    legend(ax,'show','Location','best','FontSize',10);
    if decision
        if strcmp(kind,'logisticInitial'), label = 'Initial P(Y=1|x) = 0.5';
        else, label = [model.name ' decision boundary']; end
        if D.scenario==3, label = [label ' (central region)']; end
        title(ax,label,'FontSize',12);
    elseif strcmp(kind,'pooled')
        title(ax,'LDA: one shared covariance');
    elseif strcmp(kind,'qdaCov')
        title(ax,'QDA: two class covariances');
    else
        title(ax,'Training observations');
    end
end

function colors = classColors()
    colors = [.08 .39 .75;.9 .34 .13];
end

function drawClasses(ax,X,y,classes,pointSize)
    colors = classColors(); hold(ax,'on');
    for k = classes
        idx = y==k;
        scatter(ax,X(idx,1),X(idx,2),pointSize,colors(k+1,:),'filled', ...
            'DisplayName',sprintf('Class %d',k));
    end
end

function drawEllipse(ax,mu,Sigma,color)
    [V,E] = eig((Sigma+Sigma.')/2);
    t = linspace(0,2*pi,220);
    % Same Gaussian contour level for every displayed covariance.
    xy = sqrt(-2*log(.05))*V*diag(sqrt(max(diag(E),0)))*[cos(t);sin(t)];
    xy = bsxfun(@plus,xy,mu.');
    plot(ax,xy(1,:),xy(2,:),'-','Color',color,'LineWidth',2.2,'HandleVisibility','off');
end

function drawProjectionDirection(ax,D,F,method,showFeet)
    hold(ax,'on');
    drawClasses(ax,D.XTrain,D.yTrain,0:1,13);
    if all(isfinite(F.direction))
        if strcmp(method,'PCA'), origin = F.mean; else, origin = [0 0]; end
        lengthScale = norm([diff(D.fullLimits(1:2)),diff(D.fullLimits(3:4))])/2;
        linePoints = bsxfun(@plus,[-lengthScale;lengthScale]*F.direction.',origin);
        if showFeet
            idx = unique(round(linspace(1,size(D.XTrain,1),18)));
            centered = bsxfun(@minus,D.XTrain(idx,:),origin);
            feet = bsxfun(@plus,(centered*F.direction)*F.direction.',origin);
            for k = 1:numel(idx)
                plot(ax,[D.XTrain(idx(k),1),feet(k,1)],[D.XTrain(idx(k),2),feet(k,2)], ...
                    '-','Color',[.65 .65 .65],'LineWidth',.7,'HandleVisibility','off');
            end
            plot(ax,feet(:,1),feet(:,2),'ko','MarkerSize',4,'HandleVisibility','off');
            plot(ax,D.XTrain(1,1),D.XTrain(1,2),'kp','MarkerSize',13, ...
                'MarkerFaceColor',[1 .9 .2],'DisplayName','Example x');
        end
        plot(ax,linePoints(:,1),linePoints(:,2),'k-','LineWidth',2.2, ...
            'DisplayName',[method ' direction']);
    else
        text(ax,.05,.95,'No unique FDA direction','Units','normalized', ...
            'VerticalAlignment','top','FontSize',12,'BackgroundColor','w');
    end
    finishAxes(ax,D.fullLimits);
end

function edges = sharedEdges(z,count)
    z = z(isfinite(z));
    if isempty(z), edges = linspace(-1,1,count+1); return; end
    lo = min(z); hi = max(z);
    padding = .05*max(hi-lo,1);
    edges = linspace(lo-padding,hi+padding,count+1);
end

function drawOneDimension(ax,z,y,edges)
    hold(ax,'on'); colors = classColors();
    if any(~isfinite(z))
        text(ax,.05,.5,'No unique FDA projection','Units','normalized');
        axis(ax,'off'); return;
    end
    for k = 0:1
        values = histcounts(z(y==k),edges,'Normalization','pdf');
        stairs(ax,edges,[values values(end)],'Color',colors(k+1,:), ...
            'LineWidth',2,'DisplayName',sprintf('Class %d',k));
    end
    xlim(ax,[edges(1) edges(end)]);
    xlabel(ax,'z (one-dimensional feature)'); ylabel(ax,'Density');
    set(ax,'FontSize',10); grid(ax,'on'); box(ax,'on');
    legend(ax,'show','Location','best','FontSize',9);
end

function [g1,g2,points] = makeGrid(limits)
    [g1,g2] = meshgrid(linspace(limits(1),limits(2),220),linspace(limits(3),limits(4),220));
    points = [g1(:),g2(:)];
end

function drawDecision(ax,model,limits)
    hold(ax,'on');
    [g1,g2,points] = makeGrid(limits);
    score = reshape(modelScore(model,points),size(g1));
    imagesc(ax,g1(1,:),g2(:,1),double(score>0),'HandleVisibility','off');
    colormap(ax,[.87 .93 1;1 .92 .85]); caxis(ax,[0 1]);
    set(ax,'YDir','normal');
    addBoundary(ax,g1,g2,score,'k-',2.2);
end

function addBoundary(ax,g1,g2,score,lineStyle,lineWidth)
    if min(score(:))<0 && max(score(:))>0
        contour(ax,g1,g2,score,[0 0],lineStyle,'LineWidth',lineWidth,'HandleVisibility','off');
    end
end

function showPointResult(ax,model,x)
    if strcmp(model.kind,'logistic')
        str = sprintf('P(Y=1|x) = %.4f\nPredicted class = %d', ...
            sigmoid(modelScore(model,x)),predictClass(model,x));
    else
        delta = discriminants(model,x);
        str = sprintf('\\delta_0(x) = %.4f\n\\delta_1(x) = %.4f\nPredicted class = %d', ...
            delta,predictClass(model,x));
    end
    text(ax,.025,.97,str,'Units','normalized','VerticalAlignment','top', ...
        'Interpreter','tex','FontSize',12,'FontWeight','bold', ...
        'BackgroundColor','w','EdgeColor',[.7 .7 .7],'Margin',6);
end

function finishAxes(ax,limits)
    set(ax,'DataAspectRatio',[1 1 1]);
    axis(ax,limits);
    set(ax,'YDir','normal','FontSize',11,'Layer','top');
    xlabel(ax,'x_1'); ylabel(ax,'x_2'); grid(ax,'on'); box(ax,'on');
end

function drawAccuracyBars(ax,accuracy)
    handles = bar(ax,100*accuracy,'grouped');
    set(ax,'XTick',1:3,'XTickLabel',{'Original 2D','PCA 1D','FDA 1D'},'FontSize',11);
    ylim(ax,[0 100]); ylabel(ax,'Test accuracy (%)'); grid(ax,'on');
    legend(ax,handles,{'LDA','QDA','Logistic'},'Location','southoutside','Orientation','horizontal');
end

function showSummary(ui,results)
    arrangeWindow(ui,'wide',true);
    delete(get(ui.plotPanel,'Children'));
    set(ui.heading,'String','Classification before and after feature extraction');
    set(ui.stepTitle,'String','Test accuracy on the same observations');
    count = numel(results);
    for j = 1:count
        width = .86/count;
        left = .065+(j-1)*(.91/count);
        ax = axes('Parent',ui.plotPanel,'Position',[left .2 width .68]);
        drawAccuracyBars(ax,results(j).accuracyByFeatures);
        title(ax,sprintf('Dataset %d',results(j).scenario));
        if count>1, set(ax,'XTickLabel',{'2D','PCA','FDA'}); end
    end
    set(ui.explanation,'String',wrapWords( ...
        'PCA keeps variance. FDA uses the class means and class spreads. Classification performance also depends on the classifier.',56));
    showFormulas(ui.formulaAxes,{});
    str = sprintf('Dataset / features      LDA      QDA   Logistic\n');
    names = {'2D','PCA','FDA'};
    for j = 1:count
        for k = 1:3
            str = [str sprintf('  %d / %-6s         %6.2f   %6.2f   %6.2f\n', ...
                results(j).scenario,names{k},100*results(j).accuracyByFeatures(k,:))]; %#ok<AGROW>
        end
    end
    set(ui.numbers,'String',str);
    set(ui.next,'Enable','off','String','Complete');
    setappdata(ui.fig,'Waiting',false);
    drawnow;
end
