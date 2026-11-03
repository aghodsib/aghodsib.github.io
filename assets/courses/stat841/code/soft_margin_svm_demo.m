function results=soft_margin_svm_demo(mode)
% SOFT_MARGIN_SVM_DEMO  Hard versus soft margin, slack, KKT, and kernels.
% Run soft_margin_svm_demo; or results=soft_margin_svm_demo('selftest').
% Requires Optimization Toolbox: quadprog and linprog. No other toolbox.
% X is d-by-n. y, alpha, and xi are n-by-1. Scalar c is the penalty
% parameter often written C. Notation: beta, beta_0, alpha_i, xi_i.
% Primal: min .5*||beta||^2+c*sum(xi), y_i*d(x_i)>=1-xi_i, xi_i>=0.
% Dual: max sum(alpha)-.5*alpha'*H*alpha, y'*alpha=0, 0<=alpha<=c.
% Hard margin uses no upper bound, not a large finite c. No ridge is added.
% Exact opposite-label duplicates certify hard infeasibility for ANY kernel.
% No-free-support-vector cases use the KKT interval for beta_0.
% Predictors keep all coefficients; numerical thresholds only label points.
if nargin<1, mode='gui'; end
if exist('quadprog','file')~=2 || exist('linprog','file')~=2
    error('softSVM:Toolbox','Optimization Toolbox (quadprog and linprog) is required.');
end
if strcmpi(mode,'selftest'), results=runTests(); return; end
if ~strcmpi(mode,'gui'), error('Use ''gui'' or ''selftest''.'); end
results=[]; dataId=2; kind=1; c=1; sigma=.6; stage=1; point=1; busy=false;
cs=[.05 .2 1 5 25 100]; widths=[.3 .6 1.2]; cache=containers.Map;
[X,y]=dataset(dataId); soft=[]; hard=[];
bg=[.97 .98 1];
f=figure('Name','Soft-margin SVM','NumberTitle','off','Color',bg, ...
    'Units','normalized','Position',[.025 .05 .95 .87]);
heading=uicontrol(f,'Style','text','Units','normalized','Position',[.04 .935 .92 .045], ...
    'BackgroundColor',bg,'FontSize',22,'FontWeight','bold','HorizontalAlignment','left');
formulaAx=axes('Parent',f,'Position',[.045 .81 .91 .11]);
a=axes('Parent',f,'Position',[.065 .35 .38 .40]);
b=axes('Parent',f,'Position',[.565 .35 .38 .40]);
note=uicontrol(f,'Style','text','Units','normalized','Position',[.05 .15 .90 .145], ...
    'FontSize',14,'BackgroundColor',bg,'HorizontalAlignment','left');
dataMenu=menu([.04 .065 .25 .04],{'Clean separated clouds','Overlapping clouds', ...
    'Conflicting labels at one location','Ring with two flipped labels'},2,@newData);
kMenu=menu([.31 .065 .17 .04],{'Linear','Quadratic','Gaussian RBF'},1,@newKernel);
cMenu=menu([.50 .065 .12 .04],arrayfun(@(z)sprintf('c = %g',z),cs,'UniformOutput',false),3,@newC);
sMenu=menu([.64 .065 .14 .04],{'sigma = 0.3','sigma = 0.6','sigma = 1.2'},2,@newWidth);
back=button([.80 .065 .07 .04],'< Back',@(~,~)move(-1));
next=button([.89 .065 .07 .04],'Next >',@(~,~)move(1));
pMenu=menu([.065 .305 .20 .035],pointLabels(),1,@newPoint);
compare=button([.745 .305 .20 .035],'Compare soft-margin kernels',@(~,~)compareAll());
labels={'Data','Kernel','Penalty','RBF width'}; pos=[.04 .31 .50 .64];
for j=1:4
    uicontrol(f,'Style','text','Units','normalized','Position',[pos(j) .108 .16 .023], ...
        'String',labels{j},'BackgroundColor',bg,'HorizontalAlignment','left','FontSize',10);
end
uicontrol(f,'Style','text','Units','normalized','Position',[.04 .015 .92 .027], ...
    'BackgroundColor',bg,'FontSize',10,'String', ...
    'Solid green: boundary. Dashed: margins. Black rings: free support vectors. Orange squares: bounded support vectors. Purple ring: inspected point.');
controls=[dataMenu kMenu cMenu sMenu back next pMenu compare]; refresh();
    function h=menu(p,strings,v,cb)
        h=uicontrol(f,'Style','popupmenu','Units','normalized','Position',p, ...
            'String',strings,'Value',v,'FontSize',11,'Callback',cb);
    end
    function h=button(p,t,cb)
        h=uicontrol(f,'Style','pushbutton','Units','normalized','Position',p, ...
            'String',t,'FontSize',11,'Callback',cb);
    end
    function s=pointLabels()
        s=arrayfun(@(i)sprintf('Inspect point %d (y = %+d)',i,y(i)),1:numel(y),'UniformOutput',false);
    end
    function newData(~,~)
        if busy, return; end
        dataId=get(dataMenu,'Value'); [X,y]=dataset(dataId); point=1; stage=1;
        set(pMenu,'String',pointLabels(),'Value',1); refresh();
    end
    function newKernel(~,~)
        if busy, return; end
        kind=get(kMenu,'Value'); refresh();
    end
    function newC(~,~)
        if busy, return; end
        c=cs(get(cMenu,'Value')); refresh();
    end
    function newWidth(~,~)
        if busy, return; end
        sigma=widths(get(sMenu,'Value')); refresh();
    end
    function newPoint(~,~)
        if busy, return; end
        point=get(pMenu,'Value'); render();
    end
    function move(d)
        if busy, return; end
        stage=min(5,max(1,stage+d)); refresh();
    end
    function m=getFit(k,penalty)
        key=sprintf('%d-%d-%.4g-%.4g',dataId,k,sigma,penalty);
        if ~isKey(cache,key), cache(key)=fitModel(X,y,k,sigma,penalty); end
        m=cache(key);
    end
    function enable()
        if ~isgraphics(f), return; end
        set(controls,'Enable','on'); set(back,'Enable',onoff(stage>1));
        set(next,'Enable',onoff(stage<5)); set(sMenu,'Enable',onoff(kind==3));
        set(pMenu,'Visible',onoff(stage==2 || stage==3));
    end
    function refresh()
        busy=true; set(controls,'Enable','off'); set(note,'String','Fitting and checking the SVM solutions...'); drawnow;
        try
            soft=getFit(kind,c); hard=getFit(kind,Inf); render();
        catch err
            if isgraphics(f), set(note,'String',err.message); end
            warning('softSVM:Operation','%s',err.message);
        end
        busy=false; enable();
    end
    function equations(lines)
        cla(formulaAx); axis(formulaAx,'off');
        for j=1:numel(lines)
            text(formulaAx,0,1-(j-1)*.47,lines{j},'Interpreter','latex', ...
                'Units','normalized','FontSize',16,'VerticalAlignment','top');
        end
    end
    function render()
        colorbar(a,'off'); colorbar(b,'off'); cla(a,'reset'); cla(b,'reset');
        set(a,'Position',[.065 .35 .38 .40]); set(b,'Position',[.565 .35 .38 .40]);
        titles={'Hard margin versus soft margin','Slack measures a margin violation', ...
            'Two kinds of soft-margin support vectors','The penalty changes the compromise', ...
            'Soft margin and kernels solve different problems'};
        set(heading,'String',sprintf('%d / 5    %s',stage,titles{stage}));
        primal='$\min_{\boldsymbol{\beta},\beta_0,\boldsymbol{\xi}}\ \frac12\|\boldsymbol{\beta}\|^2+c\sum_i\xi_i,\quad y_i d(\mathbf{x}_i)\geq1-\xi_i,\quad\xi_i\geq0$';
        pred='$d(\mathbf{x})=\sum_i\alpha_i y_i k(\mathbf{x}_i,\mathbf{x})+\beta_0$';
        equations({primal,pred});
        if stage==1 || stage==5
            plotModel(a,X,y,hard,0); title(a,{'Hard margin',statusText(hard)},'FontSize',12);
            plotModel(b,X,y,soft,0); title(b,{sprintf('Soft margin, c = %g',c),statusText(soft)},'FontSize',12);
            if stage==1
                lines={'Hard margin permits no violations: every y_i d(x_i) must be at least 1.', ...
                    'Soft margin penalizes violations while keeping the coefficient vector small.', ...
                    'Solid green: d = 0. Dashed blue/red: d = -1/+1. Class colours never change.'};
                if dataId==3
                    lines{3}='The two points at the origin have opposite labels. No kernel can classify both correctly.';
                end
            else
                lines={'Soft margin allows violations; a kernel changes the shape of the decision boundary.', ...
                    'Increasing c cannot turn a linear classifier into a nonlinear one.', ...
                    'Try the noisy ring with RBF, then vary c. Fitting every training label can mean fitting noise.'};
            end
        elseif ~soft.ok
            plotModel(a,X,y,soft,0); axis(b,'off'); lines={soft.message};
        elseif stage==2
            equations({primal,'$\xi_i=\max(0,1-y_i d(\mathbf{x}_i))$'});
            plotModel(a,X,y,soft,point); title(a,'Inspect a point and its signed score');
            bar(b,1:numel(y),soft.xi,'FaceColor',[.5 .58 .72]); hold(b,'on');
            bar(b,point,soft.xi(point),'FaceColor',[.9 .55 .08]);
            yline(b,1,'k--','xi = 1'); xlabel(b,'Observation i'); ylabel(b,'\xi_i'); grid(b,'on');
            title(b,'Slack is not simply the number of errors');
            lines={pointText(soft,point), ...
                'xi = 0: on/outside the margin.  0 < xi < 1: inside the margin, correctly classified.', ...
                'xi = 1: on the decision boundary.  xi > 1: misclassified. Slack is dimensionless.'};
        elseif stage==3
            equations({'$\max_{\boldsymbol{\alpha}}\ \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_i y_j k(\mathbf{x}_i,\mathbf{x}_j),\quad 0\leq\alpha_i\leq c,\quad\sum_i\alpha_i y_i=0$', ...
                '$\alpha_i[y_i d(\mathbf{x}_i)-1+\xi_i]=0,\qquad(c-\alpha_i)\xi_i=0$'});
            plotModel(a,X,y,soft,point); title(a,'Black rings: free SVs. Orange squares: bounded SVs');
            stem(b,1:numel(y),soft.alpha,'Marker','none','Color',[.3 .3 .35]); hold(b,'on');
            yline(b,c,'--','c'); scatter(b,point,soft.alpha(point),80,[.9 .55 .08],'filled');
            xlabel(b,'Observation i'); ylabel(b,'\alpha_i'); grid(b,'on'); title(b,'The dual coefficient is capped at c');
            lines={'Hard-margin support vectors lie on the margin. Soft-margin support vectors can also lie inside it or be misclassified.', ...
                '0 < alpha < c: on the margin, xi = 0.  alpha = c: on/inside the margin or misclassified.', ...
                pointText(soft,point)};
        else
            plotModel(a,X,y,soft,0); title(a,sprintf('Current fit: c = %g',c));
            loss=nan(size(cs)); sizeBeta=loss;
            for j=1:numel(cs)
                m=getFit(kind,cs(j)); if m.ok, loss(j)=sum(m.xi); sizeBeta(j)=m.norm2; end
            end
            yyaxis(b,'left'); semilogx(b,cs,loss,'o-','LineWidth',1.6); ylabel(b,'Total slack');
            yyaxis(b,'right'); semilogx(b,cs,sizeBeta,'s-','LineWidth',1.6); ylabel(b,'||\beta||^2');
            xlabel(b,'Penalty c'); grid(b,'on'); title(b,'Same observations and kernel at every c');
            lines={'Small c gives violations less weight. Large c gives violations more weight.', ...
                'At exact optima, total slack does not increase as c increases; ||beta||^2 does not decrease.', ...
                'The number of errors need not change monotonically. c and sigma control different aspects of the fit.'};
        end
        if soft.ok
            lines{end+1}=sprintf('Soft fit: %d wrong, %d boundary ties, %d support vectors; total slack %.3f.', ...
                sum(soft.margin < -1e-6),sum(abs(soft.margin)<=1e-6),sum(soft.sv),sum(soft.xi));
        end
        set(note,'String',lines); set([a b],'FontSize',11); drawnow;
    end
    function compareAll()
        if busy, return; end
        busy=true; set(controls,'Enable','off'); drawnow;
        try
            g=figure('Name','Soft margin: same data, different kernels','NumberTitle','off', ...
                'Color','w','Units','normalized','Position',[.035 .14 .93 .68]);
            names={'Linear','Quadratic','Gaussian RBF'};
            for k=1:3
                ax=subplot(1,3,k,'Parent',g); m=getFit(k,c); plotModel(ax,X,y,m,0);
                title(ax,{names{k},statusText(m)},'FontSize',12);
            end
            annotation(g,'textbox',[.05 .015 .90 .07],'String', ...
                sprintf('c = %g; RBF sigma = %.2f. Black rings: 0 < alpha < c. Orange squares: alpha = c. Kernel scales differ, so the same c is not an equal regularization strength across kernels.',c,sigma), ...
                'EdgeColor','none','FontSize',12,'HorizontalAlignment','center');
        catch err
            warning('softSVM:Comparison','%s',err.message);
        end
        busy=false; enable();
    end
end

function [X,y]=dataset(id)
j=1:24; t=j*pi*(3-sqrt(5)); r=sqrt(j/24);
U=[r.*cos(t);r.*sin(t)];
if id==1 || id==3
    X=[bsxfun(@plus,.55*U,[-1;.1]),bsxfun(@plus,.55*U,[1;-.1])];
    y=[-ones(24,1);ones(24,1)];
    if id==3, X=[X [0;0] [0;0]]; y=[y;-1;1]; end
elseif id==2
    X=[bsxfun(@plus,.85*U,[-.45;.08]),bsxfun(@plus,.85*U,[.45;-.08])];
    y=[-ones(24,1);ones(24,1)];
else
    t=2*pi*(0:31)/32; r=1.3+.12*cos(5*t);
    X=[.7*U [r.*cos(t);r.*sin(t)]];
    y=[-ones(24,1);ones(32,1)]; y([8 19])=1;
end
end
function K=kernel(A,B,k,s)
if k==1, K=A.'*B;
elseif k==2, K=(A.'*B).^2;
else
    D=bsxfun(@plus,sum(A.^2,1).',sum(B.^2,1))-2*A.'*B;
    K=exp(-max(D,0)/(2*s*s));
end
end
function Z=phi(X,k)
if k==1, Z=X; else, Z=[X(1,:).^2;X(2,:).^2;sqrt(2)*X(1,:).*X(2,:)]; end
end
function m=fitModel(X,y,k,s,c)
n=numel(y); K=kernel(X,X,k,s); K=(K+K.')/2; H=(y*y.').*K; H=(H+H.')/2;
m=struct('ok',false,'status','numerical','message','Numerical result not validated.', ...
    'kind',k,'sigma',s,'c',c,'K',K);
if isinf(c)
    for i=1:n
        if any(all(bsxfun(@eq,X,X(:,i)),1).' & y~=y(i))
            m.status='infeasible'; m.message='Opposite labels at the same location: hard margin is impossible.'; return
        end
    end
    if k<=2
        Z=phi(X,k); B=-bsxfun(@times,y,[Z.' ones(n,1)]);
        [v,~,flag]=linprog(zeros(size(B,2),1),B,-ones(n,1),[],[],[],[], ...
            optimoptions('linprog','Display','none'));
        if flag==-2, m.status='infeasible'; m.message='Hard-margin constraints are infeasible.'; return; end
        if flag<=0 || isempty(v) || max(B*v+1)>1e-5
            m.message='Hard-margin feasibility check was numerically inconclusive.'; return
        end
    end
    ub=[];
else
    ub=c*ones(n,1);
end
opts=optimoptions('quadprog','Display','none','Algorithm','interior-point-convex', ...
    'OptimalityTolerance',1e-9,'ConstraintTolerance',1e-9,'MaxIterations',1500);
[alpha,~,flag]=quadprog(H,-ones(n,1),[],[],y.',0,zeros(n,1),ub,[],opts);
if flag<=0 || isempty(alpha) || any(~isfinite(alpha))
    m.message=sprintf('QP did not converge (flag %d); this is not proof of infeasibility.',flag); return
end
tol=1e-6*max(1,max(alpha)); if isfinite(c), tol=min(tol,1e-4*c); end
zero=alpha<=tol; bound=false(n,1); if isfinite(c), bound=alpha>=c-tol; end
free=~zero & ~bound; g=K*(alpha.*y);
if any(free)
    beta0=mean(y(free)-g(free));
else
    % KKT bounds work even when every support vector is at alpha=c.
    lower=max([1-g(y>0 & zero);-1-g(y<0 & bound);-Inf]);
    upper=min([-1-g(y<0 & zero);1-g(y>0 & bound);Inf]);
    if ~isfinite(lower) || ~isfinite(upper) || lower>upper+1e-4
        m.message='Could not recover a reliable intercept from the KKT interval.'; return
    end
    beta0=(lower+upper)/2;
end
d=g+beta0; margin=y.*d; xi=max(0,1-margin); norm2=max(0,alpha.'*H*alpha);
dual=sum(alpha)-.5*norm2;
if isfinite(c)
    primal=.5*norm2+c*sum(xi);
    residual=max([max(abs(alpha.*(margin-1+xi))),max(abs((c-alpha).*xi))]);
    box=max([0;-alpha;alpha-c]);
else
    primal=.5*norm2; residual=max(abs(alpha.*(margin-1))); box=max([0;-alpha]);
    if min(margin)<1-3e-4, m.message='Hard-margin constraints failed validation.'; return; end
end
gap=abs(primal-dual)/max(1,abs(primal));
if gap>5e-4 || residual>5e-4*max(1,max(alpha)) || ...
        abs(y.'*alpha)>1e-6*max(1,sum(alpha)) || box>1e-7
    m.message='The numerical solution failed the KKT or duality-gap checks.'; return
end
m.ok=true; m.status='solved'; m.message='Verified optimum within numerical tolerance.';
m.alpha=alpha; m.beta0=beta0; m.margin=margin; m.xi=xi; m.norm2=norm2;
m.sv=~zero; m.free=free; m.bound=bound; m.gap=gap; m.residual=residual;
end
function d=predict(m,X,y,G)
d=(m.alpha.*y).'*kernel(X,G,m.kind,m.sigma)+m.beta0;
end
function plotModel(ax,X,y,m,point)
hold(ax,'on'); blue=[.1 .4 .82]; red=[.86 .22 .20];
lo=min(X,[],2); hi=max(X,[],2); center=(lo+hi)/2; half=.62*max(hi-lo);
lims=[center(1)-half center(1)+half center(2)-half center(2)+half];
if m.ok
    [u,v]=meshgrid(linspace(lims(1),lims(2),140),linspace(lims(3),lims(4),140));
    F=reshape(predict(m,X,y,[u(:).';v(:).']),size(u));
    contourf(ax,u,v,double(F>0),[-.5 .5 1.5],'LineStyle','none');
    colormap(ax,[.91 .95 1;1 .94 .93]); caxis(ax,[0 1]);
    if min(F(:))<0 && max(F(:))>0, contour(ax,u,v,F,[0 0],'Color',[.1 .53 .32],'LineWidth',2); end
    if min(F(:))<-1 && max(F(:))>-1, contour(ax,u,v,F,[-1 -1],'--','Color',blue,'LineWidth',1.1); end
    if min(F(:))<1 && max(F(:))>1, contour(ax,u,v,F,[1 1],'--','Color',red,'LineWidth',1.1); end
end
h1=scatter(ax,X(1,y<0),X(2,y<0),42,blue,'filled');
h2=scatter(ax,X(1,y>0),X(2,y>0),50,red,'x','LineWidth',1.6);
if m.ok
    scatter(ax,X(1,m.free),X(2,m.free),120,'k','o','LineWidth',1.2);
    scatter(ax,X(1,m.bound),X(2,m.bound),125,[.95 .53 .04],'s','LineWidth',1.4);
end
if point>0
    scatter(ax,X(1,point),X(2,point),225,[.6 .1 .65],'o','LineWidth',2);
    text(ax,X(1,point)+.04,X(2,point)+.06,sprintf('i=%d',point),'FontSize',11,'Color',[.5 .05 .6]);
end
axis(ax,'equal'); axis(ax,lims); grid(ax,'on'); box(ax,'on');
xlabel(ax,'x_1'); ylabel(ax,'x_2'); legend(ax,[h1 h2],{'y = -1','y = +1'}, ...
    'Location','northoutside','Orientation','horizontal');
end
function s=pointText(m,i)
if m.bound(i), type='bounded support vector'; elseif m.free(i), type='free support vector'; else, type='not a support vector'; end
s=sprintf('Point %d: y_i d(x_i) = %.3f, xi_i = %.3f, alpha_i = %.3f; %s.',i,m.margin(i),m.xi(i),m.alpha(i),type);
end
function s=statusText(m)
if m.ok
    s=sprintf('%d wrong, %d ties; %d support vectors',sum(m.margin < -1e-6),sum(abs(m.margin)<=1e-6),sum(m.sv));
elseif strcmp(m.status,'infeasible'), s='No feasible hard-margin separator';
else, s='Numerical solution not validated'; end
end
function s=onoff(c)
if c, s='on'; else, s='off'; end
end
function results=runTests()
results=struct([]); row=0; cs=[.05 .2 1 5 25 100];
for id=1:4
    [X,y]=dataset(id);
    for k=1:3
        if k==3, widths=[.3 .6 1.2]; else, widths=.6; end
        for sigma=widths
            losses=[]; norms=[];
            for c=cs
                m=fitModel(X,y,k,sigma,c); assert(m.ok,'%s',m.message);
                assert(all(m.xi>=0) && all(m.margin+m.xi>=1-1e-8));
                losses(end+1)=sum(m.xi); norms(end+1)=m.norm2; %#ok<AGROW>
                if id==3, assert(sum(m.xi(end-1:end))>=2-1e-6); end
                if k<=2
                    Z=phi(X,k); E=Z.'*Z-m.K; assert(max(abs(E(:)))<1e-10);
                end
                row=row+1; results(row).dataset=id; results(row).kernel=k;
                results(row).sigma=sigma; results(row).c=c; results(row).gap=m.gap;
            end
            assert(all(diff(losses)<=.02)); assert(all(diff(norms)>=-.02));
            h=fitModel(X,y,k,sigma,Inf);
            if id==3, assert(strcmp(h.status,'infeasible')); end
            if id==2 && k==1, assert(strcmp(h.status,'infeasible')); end
            if id==1 && k==1, assert(h.ok); end
            fprintf('Data %d, kernel %d, sigma %.1f: soft presets pass; hard status %s\n',id,k,sigma,h.status);
        end
    end
end
% An explicit no-free-SV case tests intercept recovery at small c.
X=[-1 1;0 0]; y=[-1;1]; m=fitModel(X,y,1,.6,.05);
assert(m.ok && ~any(m.free) && all(m.bound));
fprintf('%d soft-margin presets and the no-free-SV case passed.\n',row);
end
