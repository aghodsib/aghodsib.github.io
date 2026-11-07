function result = decision_tree_demo(mode)
% DECISION_TREE_DEMO  Interactive regression and classification trees.
%   decision_tree_demo opens six scenes. Use the menu or Back/Next.
%   decision_tree_demo('selftest') checks splitting, prediction, growth,
%   impurity, and cost-complexity pruning without opening a figure.
%
% MATLAB R2016b or later. No additional toolboxes or external data required.
% Reference: STAT 841, w18stat841All-v4.pptx, slides 354-379.
% Data convention: X is d-by-n, one observation per column; y is 1-by-n
% internally. Scalar constants/proportions use lowercase c and p.
%
% Scenes and suggested actions:
% 1 Move the threshold, change the feature, and click an observation.
% 2 Move the threshold, then Find best split; compare one/two features.
% 3 Press Next split repeatedly; click an observation to trace its path.
% 4 Increase the leaf count and noise; compare training/validation error.
% 5 Increase alpha; watch branches collapse. Try Choose by validation.
% 6 Compare Gini, entropy and misclassification; try Find best split.
%
% Algorithm details:
% Splits use x_j < s (left) and x_j >= s (right). Candidate thresholds are
% midpoints between DISTINCT sorted values, subject to minimum leaf size.
% Each leaf predicts its sample mean (regression), or majority class.
% The growing sequence selects the leaf with largest available SSE decrease.
% Pruning is weakest-link cost-complexity pruning of the displayed full
% tree, with total SSE + alpha*number_of_leaves. Validation is never used
% to choose individual splits. Entropy uses natural logarithms (nats).
% The pruning optimum is relative to subtrees of this grown tree, not all
% possible trees. Left/right colors in regression mean split membership;
% blue/orange in classification always mean classes 0/1.
if nargin == 0, mode = 'gui'; end
if strcmpi(mode,'selftest'), result = selftest(); return; end
if ~strcmpi(mode,'gui'), error('Use ''gui'' or ''selftest''.'); end
D = demoData(); blue = [.12 .40 .78]; orange = [.90 .38 .10];
green = [.12 .54 .31]; purple = [.50 .25 .65]; bg = [.97 .98 1];
scene = 1; feature = 1; threshold = .5; dimensions = 1; stage = 1;
sizeIndex = 1; noise = .35; alpha = 0; metric = 'gini'; selected = 1;
smallPath = grow(D.X,D.y,8,8);
[pruned,breaks] = prunePath(smallPath{end});
leafCounts = cellfun(@(t) numel(leafIds(t)),pruned);
trainRisk = cellfun(@treeRisk,pruned);
valRisk = cellfun(@(t) mean((predict(t,D.Xv)-D.yv).^2),pruned);
alphaMax = max(1,1.1*breaks(end));
largePath={}; trainErrors=[]; validErrors=[]; rebuildLarge();
f = figure('Name','Decision trees','NumberTitle','off','Color',bg, ...
    'Units','normalized','Position',[.025 .055 .95 .86]);
result = f;
heading = label([.04 .925 .92 .045],'',22);
formula = axes('Parent',f,'Position',[.045 .845 .91 .065]);
summary = label([.045 .775 .91 .06],'',11);
a = axes('Parent',f,'Position',[.065 .365 .26 .365]);
b = axes('Parent',f,'Position',[.40 .365 .25 .365]);
c = axes('Parent',f,'Position',[.735 .365 .225 .365]);
explain = label([.045 .105 .91 .085],'',13);
nav = uicontrol(f,'Style','popupmenu','Units','normalized', ...
    'Position',[.04 .035 .35 .045],'FontSize',11, ...
    'String',{'1. A split becomes a branch','2. Choose a regression split', ...
    '3. Grow a tree','4. Overfitting','5. Prune a tree','6. Classification impurity'}, ...
    'Callback',@changeScene);
back = button([.42 .035 .10 .045],'Back',@(~,~)go(-1));
next = button([.54 .035 .10 .045],'Next',@(~,~)go(1));
button([.83 .035 .13 .045],'Reset scene',@(~,~)resetScene());
controls = gobjects(0); render();
    function h = label(pos,str,sz)
        h = uicontrol(f,'Style','text','Units','normalized','Position',pos, ...
            'String',str,'FontSize',sz,'BackgroundColor',bg,'HorizontalAlignment','left');
    end
    function h = button(pos,str,cb)
        h = uicontrol(f,'Style','pushbutton','Units','normalized','Position',pos, ...
            'String',str,'FontSize',11,'Callback',cb);
    end
    function slider(pos,lo,hi,value,caption,cb,discrete)
        h = label([pos(1) .275 pos(3) .035],caption,11); controls(end+1)=h;
        h = uicontrol(f,'Style','slider','Units','normalized','Position',pos, ...
            'Min',lo,'Max',hi,'Value',min(hi,max(lo,value)),'Callback',cb);
        if discrete, set(h,'SliderStep',[1/max(1,hi-lo) min(1,5/max(1,hi-lo))]); end
        controls(end+1)=h;
    end
    function popup(pos,items,value,caption,cb)
        controls(end+1)=label([pos(1) .275 pos(3) .035],caption,11);
        controls(end+1)=uicontrol(f,'Style','popupmenu','Units','normalized', ...
            'Position',pos,'String',items,'Value',value,'FontSize',11,'Callback',cb);
    end
    function controlButton(pos,str,cb)
        controls(end+1)=button(pos,str,cb);
    end
    function go(delta)
        scene=max(1,min(6,scene+delta)); render();
    end
    function changeScene(src,~), scene=get(src,'Value'); render(); end
    function resetScene()
        switch scene
            case {1,6}, feature=1; threshold=.5; selected=1; metric='gini';
            case 2, dimensions=1; feature=1; threshold=.5;
            case 3, stage=1; selected=1;
            case 4, sizeIndex=1; noise=.35; rebuildLarge();
            case 5, alpha=0; selected=1;
        end
        render();
    end
    function changeFeature(src,~), feature=get(src,'Value'); render(); end
    function changeThreshold(src,~), threshold=get(src,'Value'); render(); end
    function changeDimensions(src,~)
        value=get(src,'Value'); dimensions=1+(value>1); feature=max(1,value-1); render();
    end
    function changeMetric(src,~)
        names={'gini','entropy','misclassification'}; metric=names{get(src,'Value')}; render();
    end
    function changeSize(src,~), sizeIndex=round(get(src,'Value')); render(); end
    function changeNoise(src,~), noise=get(src,'Value'); rebuildLarge(); render(); end
    function changeAlpha(src,~), alpha=get(src,'Value'); render(); end
    function stepTree(delta), stage=max(1,min(numel(smallPath),stage+delta)); render(); end
    function rebuildLarge()
        trainY=D.f1+noise*D.e1; validY=D.fv1+noise*D.ev1;
        largePath=grow(D.x1,trainY,32,2);
        trainErrors=cellfun(@(t) mean((predict(t,D.x1)-trainY).^2),largePath);
        validErrors=cellfun(@(t) mean((predict(t,D.xv1)-validY).^2),largePath);
        sizeIndex=min(sizeIndex,numel(largePath));
    end
    function best()
        if scene==6
            bestSplit=splitSearch(D.X,D.classes,5,metric);
        elseif dimensions==1
            bestSplit=splitSearch(D.x1,D.f1+.18*D.e1,4,'regression');
        else
            bestSplit=splitSearch(D.X,D.y,8,'regression');
        end
        if bestSplit.j>0, feature=bestSplit.j; threshold=bestSplit.s; end
        render();
    end
    function chooseValidation()
        if scene==4
            [~,sizeIndex]=min(validErrors);
        else
            [~,k]=min(valRisk);
            if k==1, alpha=0;
            elseif k==numel(pruned), alpha=alphaMax;
            else, alpha=(breaks(k)+breaks(k+1))/2; end
        end
        render();
    end
    function clickPoint(~,~)
        pt=get(a,'CurrentPoint'); [~,selected]=min(sum((D.X-pt(1,1:2).').^2,1)); render();
    end
    function eq(str)
        cla(formula); axis(formula,'off');
        text(formula,0,.5,str,'Interpreter','latex','FontSize',17,'Units','normalized');
    end
    function featureControls()
        popup([.05 .225 .22 .04],{'Feature x_1','Feature x_2'},feature,'Split feature j',@changeFeature);
        slider([.31 .232 .32 .026],.05,.95,threshold,sprintf('Threshold s = %.3f',threshold),@changeThreshold,false);
    end
    function render()
        delete(controls(isgraphics(controls))); controls=gobjects(0);
        delete(findall(f,'Type','colorbar')); delete(findall(f,'Type','legend'));
        set(a,'Position',[.065 .365 .26 .365]);
        set(b,'Position',[.40 .365 .25 .365]);
        set(c,'Position',[.735 .365 .225 .365]);
        for ax=[a b c]
            cla(ax,'reset'); hold(ax,'on'); grid(ax,'on'); set(ax,'FontSize',10,'Box','on');
        end
        set(a,'ButtonDownFcn',[]); set(nav,'Value',scene);
        set(back,'Enable',yesno(scene>1)); set(next,'Enable',yesno(scene<6));
        names={'A split becomes a branch','Choose the feature and threshold', ...
            'Build regions one split at a time','A larger tree can fit the noise', ...
            'Simplify the tree with a complexity penalty','Choose splits using class impurity'};
        set(heading,'String',sprintf('%d / 6     %s',scene,names{scene}));
        switch scene
            case 1, splitScene();
            case 2, regressionScene();
            case 3, growthScene();
            case 4, overfitScene();
            case 5, pruningScene();
            case 6, classificationScene();
        end
        drawnow;
    end
    function splitScene()
        eq('$x_j<s:\ \mathrm{left},\qquad x_j\geq s:\ \mathrm{right}$');
        featureControls();
        T=stump(D.X,D.classes,feature,threshold,'classification');
        showRegions(a,T,D.X,D.classes,true); makeClickable();
        drawTree(b,T,D.X(:,selected),true); title(b,'Follow the selected observation');
        classBars(c,D.classes,D.X(feature,:)<threshold); title(c,'Class proportions in each child');
        [pred,leaf]=predict(T,D.X(:,selected));
        scatter(a,D.X(1,selected),D.X(2,selected),110,'k','o','LineWidth',2,'HitTest','off');
        title(a,'Data and the two prediction regions');
        set(summary,'String',sprintf('Selected observation %d: x_%d = %.3f. It reaches leaf %d, which predicts class %d.', ...
            selected,feature,D.X(feature,selected),find(leafIds(T)==leaf),pred));
        set(explain,'String',{'Move s or choose another feature. Click a data point to follow its path.', ...
            'Each leaf predicts its majority class. Blue always means class 0; orange always means class 1.'});
    end
    function regressionScene()
        eq('$\min_{j,s}\left[\sum_{\mathbf{x}_i\in R_1}(y_i-c_1)^2+\sum_{\mathbf{x}_i\in R_2}(y_i-c_2)^2\right],\quad c_m=\mathrm{mean}_{\mathbf{x}_i\in R_m} y_i$');
        choice=1; if dimensions==2, choice=feature+1; end
        popup([.05 .225 .22 .04],{'One input feature','Two features: split x_1','Two features: split x_2'},choice,'Regression example / feature',@changeDimensions);
        slider([.31 .232 .32 .026],.05,.95,threshold,sprintf('Threshold s = %.3f',threshold),@changeThreshold,false);
        controlButton([.72 .22 .23 .045],'Find best split',@(~,~)best());
        if dimensions==1, X=D.x1; y=D.f1+.18*D.e1; feature=1;
        else, X=D.X; y=D.y; end
        mask=X(feature,:)<threshold; [cost,means]=splitCost(y,mask,'regression');
        [~,tables]=splitSearch(X,y,4,'regression');
        scatter(a,X(feature,mask),y(mask),22,blue,'filled'); scatter(a,X(feature,~mask),y(~mask),22,orange,'filled');
        drawResiduals(a,X(feature,:),y,mask,means);
        plot(a,[0 threshold],[means(1) means(1)],'Color',blue,'LineWidth',3);
        plot(a,[threshold 1],[means(2) means(2)],'Color',orange,'LineWidth',3);
        vertical(a,threshold,'k--'); xlim(a,[0 1]); xlabel(a,sprintf('x_%d',feature)); ylabel(a,'y'); title(a,'Each child predicts its mean');
        for j=1:size(X,1), plot(b,tables{j}(:,1),tables{j}(:,2),'LineWidth',1.8); end
        scatter(b,threshold,cost,70,'k','filled'); xlabel(b,'Candidate threshold s'); ylabel(b,'Total squared error'); title(b,'Search the possible splits');
        if size(X,1)==1, legend(b,{'Feature x_1','Current split'},'Location','best');
        else, legend(b,{'Feature x_1','Feature x_2','Current split'},'Location','best'); end
        if dimensions==1
            vals=[sum((y(mask)-means(1)).^2) sum((y(~mask)-means(2)).^2) cost];
            bar(c,vals,'FaceColor',purple); set(c,'XTick',1:3,'XTickLabel',{'Left','Right','Total'});
            ylabel(c,'Squared error'); title(c,'Add the two child errors');
        else
            T=stump(X,y,feature,threshold,'regression'); showRegions(c,T,X,y,false); title(c,'The split in input space');
        end
        set(summary,'String',sprintf('Left: n = %d, c_1 = %.2f. Right: n = %d, c_2 = %.2f. Total squared error = %.3f.',sum(mask),means(1),sum(~mask),means(2),cost));
        set(explain,'String',{'Move s and watch the residuals. A good split makes responses close to the mean in each child.', ...
            'Find best split compares valid thresholds across all available features. The curves use at least 4 points per child.'});
    end
    function growthScene()
        T=smallPath{stage}; count=numel(leafIds(T));
        eq('$\hat y(\mathbf{x})=c_m\quad\mathrm{when}\ \mathbf{x}\in R_m,\qquad c_m=\mathrm{mean}_{\mathbf{x}_i\in R_m}y_i$');
        controlButton([.06 .225 .24 .045],'Undo split',@(~,~)stepTree(-1));
        set(controls(end),'Enable',yesno(stage>1));
        controlButton([.35 .225 .24 .045],'Next split',@(~,~)stepTree(1));
        set(controls(end),'Enable',yesno(stage<numel(smallPath)));
        showRegions(a,T,D.X,D.y,false); makeClickable(); title(a,'Regions and observed responses');
        drawTree(b,T,D.X(:,selected),false); title(b,'The same regions as leaves');
        predictionSurface(c,T); title(c,'A constant prediction in each region');
        if stage==1, msg='Start with one leaf. Its prediction is the mean of all responses.';
        else, node=T.nodes(T.lastSplit); msg=sprintf('Latest split: x_%d < %.3f, within one existing region.',node.j,node.s); end
        set(summary,'String',sprintf('%d leaves. Training squared error = %.3f. %s',count,treeRisk(T),msg));
        set(explain,'String',{'Next split chooses the available leaf split with the largest reduction in squared error.', ...
            'A split affects only its own region. Click a point to trace its path through the tree.'});
    end
    function overfitScene()
        T=largePath{sizeIndex}; count=numel(leafIds(T)); [~,bestIndex]=min(validErrors);
        eq('$\mathrm{mean\ squared\ error}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2$');
        slider([.05 .232 .28 .026],1,numel(largePath),sizeIndex,sprintf('Number of leaves = %d',count),@changeSize,true);
        slider([.39 .232 .25 .026],.05,.8,noise,sprintf('Noise standard deviation = %.2f',noise),@changeNoise,false);
        controlButton([.72 .22 .23 .045],'Choose by validation',@(~,~)chooseValidation());
        gx=linspace(0,1,1000); fit=predict(T,gx);
        scatter(a,D.x1,D.f1+noise*D.e1,20,blue,'filled'); plot(a,gx,true1(gx),'k:','LineWidth',1.5); plot(a,gx,fit,'Color',orange,'LineWidth',2);
        title(a,'Fit to the training observations'); xlabel(a,'x_1'); ylabel(a,'y'); xlim(a,[0 1]);
        legend(a,{'Training data','True mean','Tree prediction'},'Location','best','FontSize',9);
        scatter(b,D.xv1,D.fv1+noise*D.ev1,8,[.60 .65 .70],'filled'); plot(b,gx,true1(gx),'k:','LineWidth',1.5); plot(b,gx,fit,'Color',orange,'LineWidth',2);
        title(b,'The same tree on validation data'); xlabel(b,'x_1'); ylabel(b,'y'); xlim(b,[0 1]);
        lim=[min([D.f1+noise*D.e1 D.fv1+noise*D.ev1])-.1 max([D.f1+noise*D.e1 D.fv1+noise*D.ev1])+.1]; ylim(a,lim); ylim(b,lim);
        plot(c,1:numel(largePath),trainErrors,'Color',blue,'LineWidth',2);
        plot(c,1:numel(largePath),validErrors,'Color',orange,'LineWidth',2);
        scatter(c,bestIndex,validErrors(bestIndex),70,green,'filled'); vertical(c,sizeIndex,'k--');
        xlabel(c,'Number of leaves'); ylabel(c,'Mean squared error'); title(c,'More leaves do not always help');
        legend(c,{'Training','Validation','Best validation size','Current size'},'Location','best','FontSize',9);
        set(summary,'String',sprintf('%d leaves: training MSE %.3f, validation MSE %.3f. Best validation size: %d leaves.',count,trainErrors(sizeIndex),validErrors(sizeIndex),bestIndex));
        set(explain,'String',{'Increase the leaf count. Small regions can start fitting individual noise fluctuations.', ...
            'The validation observations never determine the splits. Their error helps choose the tree size.'});
    end
    function pruningScene()
        total=trainRisk+alpha*leafCounts; [~,k]=min(total); T=pruned{k};
        eq('$\sum_{m=1}^{|T|}n_m q_m+\alpha|T|,\qquad q_m=\frac{1}{n_m}\sum_{\mathbf{x}_i\in R_m}(y_i-c_m)^2$');
        slider([.05 .232 .46 .026],0,alphaMax,alpha,sprintf('Complexity penalty alpha = %.3f',alpha),@changeAlpha,false);
        controlButton([.66 .22 .29 .045],'Choose by validation',@(~,~)chooseValidation());
        showRegions(a,T,D.X,D.y,false); makeClickable(); title(a,'Regions merge as branches collapse');
        drawTree(b,T,D.X(:,selected),false); title(b,'Selected subtree');
        [counts,order]=sort(leafCounts);
        plot(c,counts,trainRisk(order),'.-','Color',blue,'LineWidth',1.6);
        plot(c,counts,alpha*counts,'.-','Color',orange,'LineWidth',1.6);
        plot(c,counts,total(order),'.-','Color',green,'LineWidth',2);
        scatter(c,leafCounts(k),total(k),85,'k','filled'); xlabel(c,'Number of leaves'); ylabel(c,'Cost');
        title(c,'Choose the smallest total cost'); legend(c,{'Training SSE','alpha times leaves','Total','Selected'},'Location','best','FontSize',9);
        set(summary,'String',sprintf('%d leaves: SSE %.3f + penalty %.3f = %.3f. Validation MSE = %.3f.',leafCounts(k),trainRisk(k),alpha*leafCounts(k),total(k),valRisk(k)));
        set(explain,'String',{'Increase alpha. The selected tree changes when removing a branch saves enough complexity cost.', ...
            'The dots are subtrees on the weakest-link pruning path. Validation chooses among these subtrees.'});
    end
    function classificationScene()
        featureControls();
        names={'gini','entropy','misclassification'}; chosen=find(strcmp(metric,names));
        popup([.69 .225 .26 .04],{'Gini impurity','Cross-entropy (nats)','Misclassification error'},chosen,'Split criterion',@changeMetric);
        controlButton([.67 .035 .14 .045],'Find best split',@(~,~)best());
        eq('$\mathrm{split\ score}=\frac{n_1}{n}q_1+\frac{n_2}{n}q_2,\qquad \mathrm{gain}=q_{\mathrm{parent}}-\mathrm{split\ score}$');
        mask=D.X(feature,:)<threshold; T=stump(D.X,D.classes,feature,threshold,'classification');
        showRegions(a,T,D.X,D.classes,true); title(a,'Try a split in the feature space');
        classBars(b,D.classes,mask); title(b,'How mixed are the child nodes?');
        p=linspace(0,1,301);
        plot(c,p,impurity(p,'gini'),'Color',blue,'LineWidth',1.5);
        plot(c,p,impurity(p,'entropy'),'Color',orange,'LineWidth',1.5);
        plot(c,p,impurity(p,'misclassification'),'Color',purple,'LineWidth',1.5);
        proportions=[mean(D.classes) mean(D.classes(mask)) mean(D.classes(~mask))];
        scatter(c,proportions,impurity(proportions,metric),65,'k','filled');
        text(c,proportions,impurity(proportions,metric)+.035,{'parent','left','right'},'FontSize',9,'HorizontalAlignment','center');
        xlim(c,[0 1]); ylim(c,[0 .8]); xlabel(c,'Proportion p of class 1'); ylabel(c,'Impurity'); title(c,'Pure nodes have zero impurity');
        legend(c,{'Gini','Entropy','Misclassification','Current nodes'},'Location','south','FontSize',9);
        score=splitCost(D.classes,mask,metric)/numel(D.classes); parent=impurity(mean(D.classes),metric);
        set(summary,'String',sprintf('%s: parent impurity %.3f; weighted child impurity %.3f; gain %.3f. Child sizes: %d and %d.',metric,parent,score,parent-score,sum(mask),sum(~mask)));
        set(explain,'String',{'A useful split makes the children less mixed. Their impurities are weighted by their sample counts.', ...
            'Compare the criteria, then Find best split. Their numerical scales differ; compare candidate splits within one criterion.'});
    end
    function makeClickable()
        set(a,'ButtonDownFcn',@clickPoint);
        kids=get(a,'Children'); set(kids,'HitTest','off');
    end
end

function D=demoData()
r=RandStream('mt19937ar','Seed',841);
D.X=.015+.97*rand(r,2,180); D.Xv=.015+.97*rand(r,2,450);
D.X(:,1:4)=[.02 .02 .98 .98;.02 .98 .02 .98];
D.y=true2(D.X)+.35*randn(r,1,size(D.X,2));
D.yv=true2(D.Xv)+.35*randn(r,1,size(D.Xv,2));
D.classes=double((D.X(1,:)<.42 & D.X(2,:)<.68) | (D.X(1,:)>.72 & D.X(2,:)>.40));
flip=rand(r,1,size(D.X,2))<.08; D.classes(flip)=1-D.classes(flip);
D.x1=linspace(.01,.99,80); D.xv1=linspace(.002,.998,401);
D.f1=true1(D.x1); D.fv1=true1(D.xv1);
D.e1=randn(r,1,numel(D.x1)); D.ev1=randn(r,1,numel(D.xv1));
end
function y=true1(x)
y=.35*sin(2*pi*x)+.95*(x>=.32)-.85*(x>=.70);
end
function y=true2(X)
y=.3+1.2*(X(1,:)>=.42)-.9*(X(2,:)>=.62)+.7*(X(1,:)>=.73 & X(2,:)<.62);
end
function n=newNode(X,y,idx,lo,hi,depth,kind)
n=struct('idx',idx,'lo',lo,'hi',hi,'depth',depth,'j',0,'s',NaN, ...
    'left',0,'right',0,'value',0,'cost',0,'n',numel(idx));
if isempty(idx), return; end
if strcmp(kind,'classification'), n.value=double(mean(y(idx))>.5);
else, n.value=mean(y(idx)); end
n.cost=sum((y(idx)-mean(y(idx))).^2);
end
function T=rootTree(X,y,kind)
T=struct('nodes',newNode(X,y,1:numel(y),zeros(size(X,1),1),ones(size(X,1),1),0,kind),'lastSplit',0);
end
function T=splitNode(T,id,X,y,j,s,kind)
n=T.nodes(id); mask=X(j,n.idx)<s; l=numel(T.nodes)+1; r=l+1;
hi=n.hi; hi(j)=s; lo=n.lo; lo(j)=s;
T.nodes(l)=newNode(X,y,n.idx(mask),n.lo,hi,n.depth+1,kind);
T.nodes(r)=newNode(X,y,n.idx(~mask),lo,n.hi,n.depth+1,kind);
T.nodes(id).j=j; T.nodes(id).s=s; T.nodes(id).left=l; T.nodes(id).right=r; T.lastSplit=id;
end
function T=stump(X,y,j,s,kind)
T=rootTree(X,y,kind); T=splitNode(T,1,X,y,j,s,kind);
end
function ids=activeIds(T,id)
if nargin<2, id=1; end
n=T.nodes(id); ids=id;
if n.left>0, ids=[ids activeIds(T,n.left) activeIds(T,n.right)]; end
end
function ids=leafIds(T,id)
if nargin<2, id=1; end
n=T.nodes(id);
if n.left==0, ids=id; else, ids=[leafIds(T,n.left) leafIds(T,n.right)]; end
end
function risk=treeRisk(T)
ids=leafIds(T); risk=sum([T.nodes(ids).cost]);
end
function [values,terminal]=predict(T,X)
values=zeros(1,size(X,2)); terminal=values;
for i=1:size(X,2)
    id=1;
    while T.nodes(id).left>0
        n=T.nodes(id);
        if X(n.j,i)<n.s, id=n.left; else, id=n.right; end
    end
    terminal(i)=id; values(i)=T.nodes(id).value;
end
end
function [cost,means]=splitCost(y,mask,kind)
means=[NaN NaN]; cost=0; masks={mask,~mask};
for k=1:2
    v=y(masks{k});
    if isempty(v), cost=Inf; return; end
    means(k)=mean(v);
    if strcmp(kind,'regression'), cost=cost+sum((v-means(k)).^2);
    else, cost=cost+numel(v)*impurity(means(k),kind); end
end
end
function q=impurity(p,kind)
switch kind
    case 'gini', q=2*p.*(1-p);
    case 'misclassification', q=min(p,1-p);
    case 'entropy'
        q=zeros(size(p)); valid=p>0 & p<1;
        q(valid)=-p(valid).*log(p(valid))-(1-p(valid)).*log(1-p(valid));
    otherwise, error('Unknown impurity criterion.');
end
end
function [best,tables]=splitSearch(X,y,minLeaf,kind)
best=struct('j',0,'s',NaN,'cost',Inf); tables=cell(1,size(X,1));
for j=1:size(X,1)
    values=unique(X(j,:)); candidates=(values(1:end-1)+values(2:end))/2;
    tab=zeros(0,2);
    for s=candidates(:).'
        mask=X(j,:)<s;
        if sum(mask)<minLeaf || sum(~mask)<minLeaf, continue; end
        cost=splitCost(y,mask,kind); tab(end+1,:)=[s cost]; %#ok<AGROW>
        if cost<best.cost-1e-12, best=struct('j',j,'s',s,'cost',cost); end
    end
    tables{j}=tab;
end
end
function path=grow(X,y,maxLeaves,minLeaf)
T=rootTree(X,y,'regression'); path={T};
while numel(leafIds(T))<maxLeaves
    gain=1e-10; chosen=0; best=[];
    for id=leafIds(T)
        n=T.nodes(id); trial=splitSearch(X(:,n.idx),y(n.idx),minLeaf,'regression');
        if trial.j>0 && n.cost-trial.cost>gain
            gain=n.cost-trial.cost; chosen=id; best=trial;
        end
    end
    if chosen==0, break; end
    T=splitNode(T,chosen,X,y,best.j,best.s,'regression'); path{end+1}=T; %#ok<AGROW>
end
end
function [path,breaks]=prunePath(T)
% Effective alpha = SSE increase / number of removed leaves.
path={T}; breaks=0;
while T.nodes(1).left>0
    ids=activeIds(T); candidates=[]; effective=[];
    for id=ids
        if T.nodes(id).left==0, continue; end
        leaves=leafIds(T,id);
        candidates(end+1)=id; %#ok<AGROW>
        effective(end+1)=(T.nodes(id).cost-sum([T.nodes(leaves).cost]))/(numel(leaves)-1); %#ok<AGROW>
    end
    weakest=min(effective); tol=1e-10*max(1,abs(weakest));
    tied=candidates(effective<=weakest+tol);
    % Preorder visits an ancestor before its descendants, so tied ancestor
    % collapses suppress any now-inactive descendants.
    for id=tied
        if any(activeIds(T)==id), T.nodes(id).left=0; T.nodes(id).right=0; end
    end
    path{end+1}=T; breaks(end+1)=max(0,weakest); %#ok<AGROW>
end
end
function drawResiduals(ax,x,y,mask,means)
pred=means(2)*ones(size(y)); pred(mask)=means(1);
line(ax,[x;x],[y;pred],'Color',[.72 .74 .78],'HandleVisibility','off');
end
function vertical(ax,s,style)
lim=ylim(ax); plot(ax,[s s],lim,style,'LineWidth',1.3); ylim(ax,lim);
end
function classBars(ax,y,mask)
p=[mean(y(mask)) mean(y(~mask))]; h=bar(ax,[1-p(:) p(:)],'stacked');
h(1).FaceColor=[.12 .40 .78]; h(2).FaceColor=[.90 .38 .10];
set(ax,'XTick',[1 2],'XTickLabel',{sprintf('Left (n=%d)',sum(mask)),sprintf('Right (n=%d)',sum(~mask))});
ylim(ax,[0 1]); ylabel(ax,'Class proportion'); legend(ax,{'Class 0','Class 1'},'Location','northoutside','Orientation','horizontal','FontSize',9);
end
function showRegions(ax,T,X,y,classification)
leaves=leafIds(T); colors=parula(128); lim=[-.6 2.5];
for m=1:numel(leaves)
    n=T.nodes(leaves(m));
    if classification
        if n.value==0, color=[.12 .40 .78]; else, color=[.90 .38 .10]; end
    else
        index=max(1,min(128,1+round(127*(n.value-lim(1))/diff(lim)))); color=colors(index,:);
    end
    patch(ax,[n.lo(1) n.hi(1) n.hi(1) n.lo(1)],[n.lo(2) n.lo(2) n.hi(2) n.hi(2)], ...
        color,'FaceAlpha',.20,'EdgeColor',[.35 .38 .42],'LineWidth',1.3,'HandleVisibility','off');
    text(ax,n.lo(1)+.015,n.hi(2)-.035,sprintf('R_%d',m),'FontSize',10,'Color',[.1 .1 .1],'HandleVisibility','off');
end
if classification
    scatter(ax,X(1,y==0),X(2,y==0),20,[.12 .40 .78],'filled');
    scatter(ax,X(1,y==1),X(2,y==1),20,[.90 .38 .10],'filled');
    legend(ax,{'Class 0','Class 1'},'Location','northoutside','Orientation','horizontal','FontSize',9);
else
    scatter(ax,X(1,:),X(2,:),20,y,'filled'); colormap(ax,colors); caxis(ax,lim);
    cb=colorbar(ax,'southoutside'); cb.Label.String='Observed response y / region prediction c_m'; cb.FontSize=9;
end
xlabel(ax,'x_1'); ylabel(ax,'x_2'); axis(ax,[0 1 0 1]); axis(ax,'square');
end
function predictionSurface(ax,T)
colors=parula(128); lim=[-.6 2.5];
for id=leafIds(T)
    n=T.nodes(id); index=max(1,min(128,1+round(127*(n.value-lim(1))/diff(lim))));
    patch(ax,[n.lo(1) n.hi(1) n.hi(1) n.lo(1)],[n.lo(2) n.lo(2) n.hi(2) n.hi(2)], ...
        n.value*ones(1,4),colors(index,:),'FaceAlpha',.8,'EdgeColor',[.3 .3 .3]);
end
xlabel(ax,'x_1'); ylabel(ax,'x_2'); zlabel(ax,'Predicted y'); xlim(ax,[0 1]); ylim(ax,[0 1]); zlim(ax,[-.8 2.7]); view(ax,[-35 30]);
end
function drawTree(ax,T,x,classification)
ids=activeIds(T); leaves=leafIds(T); loc=zeros(1,numel(T.nodes));
for k=1:numel(leaves), loc(leaves(k))=(k-.5)/numel(leaves); end
for id=fliplr(ids)
    n=T.nodes(id); if n.left>0, loc(id)=(loc(n.left)+loc(n.right))/2; end
end
highlight=1; id=1;
while T.nodes(id).left>0
    n=T.nodes(id); if x(n.j)<n.s, id=n.left; else, id=n.right; end
    highlight(end+1)=id; %#ok<AGROW>
end
depth=max([T.nodes(ids).depth]); dy=1/max(1,depth+1);
for id=ids
    n=T.nodes(id); py=1-n.depth*dy;
    if n.left==0, continue; end
    for child=[n.left n.right]
        cy=1-T.nodes(child).depth*dy;
        if any(highlight==child), color=[.12 .54 .31]; width=2.5; else, color=[.65 .67 .70]; width=1; end
        plot(ax,[loc(id) loc(child)],[py cy],'Color',color,'LineWidth',width);
        if numel(leaves)<=4
            if child==n.left, str='yes'; else, str='no'; end
            text(ax,(loc(id)+loc(child))/2,(py+cy)/2,str,'FontSize',9,'BackgroundColor','w');
        end
    end
end
font=10; if numel(leaves)>5, font=8; end
for id=ids
    n=T.nodes(id); py=1-n.depth*dy; color=[.97 .97 .97];
    if n.left>0, str=sprintf('x_%d < %.2f',n.j,n.s);
    else
        m=find(leaves==id);
        if classification
            str=sprintf('R_%d\nclass %d\nn=%d',m,n.value,n.n);
            if n.value==0, color=[.85 .91 1]; else, color=[1 .90 .82]; end
        else, str=sprintf('R_%d\nc=%.2f',m,n.value); color=[.88 .95 .88]; end
    end
    edge=[.45 .45 .45]; if any(highlight==id), edge=[.12 .54 .31]; end
    text(ax,loc(id),py,str,'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',font,'BackgroundColor',color,'EdgeColor',edge,'Margin',3);
end
axis(ax,[-.08 1.08 -.05 1.13]); axis(ax,'off');
text(ax,.5,-.02,'Left: yes     Right: no     Green: selected path','FontSize',9,'HorizontalAlignment','center');
end
function s=yesno(v)
if v, s='on'; else, s='off'; end
end
function pairs=allSubtreeCosts(T,id)
% Exhaustive reference used ONLY by selftest, not by the pruning algorithm.
n=T.nodes(id); pairs=[n.cost 1];
if n.left==0, return; end
left=allSubtreeCosts(T,n.left); right=allSubtreeCosts(T,n.right);
for i=1:size(left,1), for j=1:size(right,1), pairs(end+1,:)=left(i,:)+right(j,:); end; end %#ok<AGROW>
end
function result=selftest()
% Known optimum; duplicate values must not produce empty-child candidates.
X=[0 0 .2 .5 .5 .8 1 1]; y=[0 0 0 2 2 2 2 2];
[b,tables]=splitSearch(X,y,2,'regression'); assert(b.j==1 && abs(b.s-.35)<1e-12 && b.cost==0);
assert(all(diff(tables{1}(:,1))>0));
T=stump(X,y,1,.35,'regression'); assert(all(predict(T,[.1 .35 .9])==[0 2 2]));
% The optimal constant is the mean, including unequal child sizes.
[c,m]=splitCost([0 2 4 10 12],[true true true false false],'regression');
assert(abs(c-10)<1e-12 && all(m==[2 11]));
for criterion={'gini','entropy','misclassification'}
    kind=criterion{1}; q=impurity([0 .25 .5 .75 1],kind);
    assert(all(isfinite(q)) && q(1)==0 && q(end)==0 && abs(q(2)-q(4))<1e-12);
    score=splitCost([0 0 1 1 1],[true true true false false],kind);
    assert(abs(score-3*impurity(1/3,kind))<1e-12);
end
D=demoData(); path=grow(D.X,D.y,8,8);
risks=cellfun(@treeRisk,path); assert(all(diff(risks)<=1e-9));
for k=1:numel(path)
    T=path{k}; ids=leafIds(T); assert(numel(ids)==k);
    assert(sum([T.nodes(ids).n])==size(D.X,2));
    assert(abs(sum((predict(T,D.X)-D.y).^2)-treeRisk(T))<1e-8);
    if k>1, assert(all([T.nodes(ids).n]>=8)); end
end
[pruned,breaks]=prunePath(path{end}); counts=cellfun(@(t) numel(leafIds(t)),pruned);
prunedRisk=cellfun(@treeRisk,pruned);
assert(all(diff(counts)<0) && counts(end)==1 && all(diff(breaks)>=-1e-8));
assert(all(diff(prunedRisk)>=-1e-8));
% Compare weakest-link path with EVERY subtree of this small full tree.
pairs=allSubtreeCosts(path{end},1);
checks=unique([0 breaks linspace(0,1.1*breaks(end),101)]);
for alpha=checks
    actual=min(prunedRisk+alpha*counts); expected=min(pairs(:,1)+alpha*pairs(:,2));
    assert(abs(actual-expected)<1e-7*max(1,abs(expected)));
end
result=struct('passed',true,'growthStages',numel(path),'pruningStages',numel(pruned), ...
    'exhaustiveSubtreesChecked',size(pairs,1),'alphaValuesChecked',numel(checks));
fprintf('Passed: splits, leaf means, predictions, weighted impurities, growth, and exhaustive pruning checks.\n');
disp(result);
end
