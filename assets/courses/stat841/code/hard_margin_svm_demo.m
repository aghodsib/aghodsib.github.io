function results = hard_margin_svm_demo(mode)
% HARD_MARGIN_SVM_DEMO  Fit and visualize genuine hard-margin kernel SVMs.
% Run: hard_margin_svm_demo
% Verify all presets without opening figures: hard_margin_svm_demo('selftest')
% Requires MATLAB with Optimization Toolbox (quadprog and linprog).
% No Statistics and Machine Learning Toolbox is needed.
%
% Notation follows the STAT 841 SVM slides: beta, beta_0, alpha_i, y_i.
% X is d-by-n, with observations in columns; y and alpha are n-by-1.
% Primal: min 0.5*||beta||^2, y_i*(beta'*phi(x_i)+beta_0) >= 1.
% Dual: max sum(alpha)-0.5*alpha'*H*alpha,
%       alpha >= 0, y'*alpha = 0; H_ij=y_i*y_j*k(x_i,x_j).
% There is NO finite box constraint C, slack penalty, or diagonal ridge.
% A failed numerical solve is not reported as proof of infeasibility.
%
% Kernels: linear; homogeneous quadratic; Gaussian RBF.
% Quadratic phi(x)=[x_1^2;x_2^2;sqrt(2)*x_1*x_2] is displayed exactly.
% RBF has an infinite-dimensional feature map. Its displayed score surface
% is d(x) over the input plane, NOT the RBF feature space.
%
% Source notation: w18stat841All-v4.pptx, slides 284 and 292-300.
% Solver docs: https://www.mathworks.com/help/optim/ug/quadprog.html
%              https://www.mathworks.com/help/optim/ug/linprog.html
if nargin<1, mode='gui'; end
if exist('quadprog','file')~=2 || exist('linprog','file')~=2
    error('svmDemo:Toolbox', ['This demo requires Optimization Toolbox ', ...
        '(quadprog and linprog). It solves the hard-margin dual directly.']);
end
if strcmpi(mode,'selftest')
    results = selftest(); return
end
if ~strcmpi(mode,'gui'), error('Use ''gui'' or ''selftest''.'); end
results = [];
blue=[.10 .40 .82]; red=[.86 .22 .20]; green=[.10 .53 .32];
bg=[.97 .98 1]; dataId=1; kind=2; sigma=.55; stage=1; busy=false;
[X,y]=dataSet(dataId); model=[]; linear=[]; cache=cell(3,5);
fig=figure('Name','Hard-margin SVM and kernels','NumberTitle','off', ...
    'Color',bg,'Units','normalized','Position',[.03 .06 .94 .86]);
heading=uicontrol(fig,'Style','text','Units','normalized', ...
    'Position',[.045 .93 .91 .045],'BackgroundColor',bg,'FontSize',22, ...
    'FontWeight','bold','HorizontalAlignment','left');
mathax=axes('Parent',fig,'Position',[.045 .82 .91 .10]);
left=axes('Parent',fig,'Position',[.07 .33 .38 .43]);
right=axes('Parent',fig,'Position',[.57 .33 .36 .43]);
notes=uicontrol(fig,'Style','text','Units','normalized', ...
    'Position',[.055 .13 .90 .13],'BackgroundColor',bg,'FontSize',14, ...
    'HorizontalAlignment','left');
label(.04,.095,.23,'Data'); label(.29,.095,.22,'Kernel');
label(.53,.095,.12,'RBF width');
dataMenu=uicontrol(fig,'Style','popupmenu','Units','normalized', ...
    'Position',[.04 .055 .23 .04],'FontSize',11, ...
    'String',{'Inner cloud and outer ring','Two curved clouds','Separated clouds'}, ...
    'Callback',@changeData);
kernelMenu=uicontrol(fig,'Style','popupmenu','Units','normalized', ...
    'Position',[.29 .055 .22 .04],'FontSize',11,'Value',2, ...
    'String',{'Linear','Quadratic polynomial','Gaussian RBF'}, ...
    'Callback',@changeKernel);
widthMenu=uicontrol(fig,'Style','popupmenu','Units','normalized', ...
    'Position',[.53 .055 .12 .04],'FontSize',11,'Value',2, ...
    'String',{'sigma = 0.25','sigma = 0.55','sigma = 1.10'}, ...
    'Callback',@changeWidth);
prev=uicontrol(fig,'Style','pushbutton','Units','normalized', ...
    'Position',[.69 .055 .10 .04],'FontSize',11,'String','< Back', ...
    'Callback',@(~,~) step(-1));
nxt=uicontrol(fig,'Style','pushbutton','Units','normalized', ...
    'Position',[.81 .055 .10 .04],'FontSize',11,'String','Next >', ...
    'Callback',@(~,~) step(1));
replay=uicontrol(fig,'Style','pushbutton','Units','normalized', ...
    'Position',[.57 .275 .17 .035],'FontSize',11,'String','Animate mapping', ...
    'Callback',@(~,~) animate());
compare=uicontrol(fig,'Style','pushbutton','Units','normalized', ...
    'Position',[.77 .275 .17 .035],'FontSize',11,'String','Compare kernels', ...
    'Callback',@(~,~) compareKernels());
controls=[dataMenu kernelMenu widthMenu prev nxt replay compare];
refresh();

    function label(x,z,w,t)
        uicontrol(fig,'Style','text','Units','normalized','Position',[x z w .025], ...
            'BackgroundColor',bg,'String',t,'HorizontalAlignment','left','FontSize',10);
    end
    function changeData(~,~)
        if busy, return; end
        dataId=get(dataMenu,'Value'); [X,y]=dataSet(dataId); stage=1; refresh();
    end
    function changeKernel(~,~)
        if busy, return; end
        kind=get(kernelMenu,'Value'); stage=max(stage,3); refresh();
    end
    function changeWidth(~,~)
        if busy, return; end
        values=[.25 .55 1.10]; sigma=values(get(widthMenu,'Value')); refresh();
    end
    function step(delta)
        if busy, return; end
        stage=min(5,max(1,stage+delta)); refresh();
        if stage==4 && delta>0 && kind==2, animate(); end
    end
    function m = fitted(k)
        index=k;
        if k==3, values=[.25 .55 1.10]; index=2+find(abs(values-sigma)<1e-8,1); end
        if isempty(cache{dataId,index})
            cache{dataId,index}=fitHardMargin(X,y,k,sigma);
        end
        m=cache{dataId,index};
    end
    function refresh()
        if ~isgraphics(fig), return; end
        busy=true; set(controls,'Enable','off');
        set(notes,'String','Solving the hard-margin optimization problem...'); drawnow;
        try
            linear=fitted(1); model=fitted(kind);
            render();
        catch err
            if isgraphics(fig)
                set(notes,'String',{'The demo could not complete this operation.',err.message});
                warning('svmDemo:Operation','%s',err.message);
            end
        end
        busy=false; enableControls();
    end
    function enableControls()
        if ~isgraphics(fig), return; end
        set(controls,'Enable','on');
        set(prev,'Enable',onoff(stage>1)); set(nxt,'Enable',onoff(stage<5));
        set(widthMenu,'Enable',onoff(kind==3));
        set(replay,'Visible',onoff(stage==4 && kind==2));
    end
    function formula(lines)
        cla(mathax); axis(mathax,'off');
        for q=1:numel(lines)
            text(mathax,0,1-(q-1)*.48,lines{q},'Units','normalized', ...
                'Interpreter','latex','FontSize',16,'VerticalAlignment','top');
        end
    end
    function render()
        colorbar(left,'off'); colorbar(right,'off');
        cla(left,'reset'); cla(right,'reset');
        set(left,'Position',[.07 .33 .38 .43]);
        set(right,'Position',[.57 .33 .36 .43]);
        titles={'Linear hard-margin SVM','Only inner products enter the dual', ...
            'Replace the inner product and refit','Geometry of the learned classifier', ...
            'Support vectors determine the classifier'};
        set(heading,'String',sprintf('%d / 5    %s',stage,titles{stage}));
        primal='$\min_{\boldsymbol{\beta},\beta_0}\ \frac12\|\boldsymbol{\beta}\|^2\quad\mathrm{subject\ to}\quad y_i(\boldsymbol{\beta}^{\top}\mathbf{x}_i+\beta_0)\geq1$';
        dual='$\max_{\boldsymbol{\alpha}}\ \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j y_i y_j k(\mathbf{x}_i,\mathbf{x}_j),\qquad \alpha_i\geq0,\quad\sum_i\alpha_i y_i=0$';
        prediction='$d(\mathbf{x})=\sum_i\alpha_i y_i k(\mathbf{x}_i,\mathbf{x})+\beta_0,\qquad \widehat y=\mathrm{sign}(d(\mathbf{x}))$';
        if stage==1
            formula({primal,'$d(\mathbf{x})=\boldsymbol{\beta}^{\top}\mathbf{x}+\beta_0$'});
            plotInput(left,X,y,linear,true,false);
            axis(right,'off');
            if linear.ok
                words={'A linear separator exists for these data.', ...
                    'The SVM chooses the largest feature-space margin.', ...
                    'Solid green: decision boundary d(x) = 0.', ...
                    'Dashed: margin lines d(x) = -1 and +1.'};
            elseif strcmp(linear.status,'infeasible')
                words={'No linear hard-margin separator exists.', ...
                    'The margin constraints are infeasible.', ...
                    'The shaded convex hulls overlap.', ...
                    'We need a different feature representation.'};
            else
                words={'The linear solver did not pass validation.',linear.message};
            end
            text(right,.02,.85,words,'Units','normalized','FontSize',15,'VerticalAlignment','top');
            set(notes,'String',{'Blue circles: y = -1. Red crosses: y = +1.', ...
                'Hard margin requires every training point to satisfy y_i d(x_i) >= 1.', ...
                'Choose Separated clouds to see an ordinary linear SVM first.'});
        elseif stage==2
            formula({strrep(dual,'k(\mathbf{x}_i,\mathbf{x}_j)','\mathbf{x}_i^{\top}\mathbf{x}_j'), ...
                '$\mathbf{x}_i^{\top}\mathbf{x}_j\quad\longrightarrow\quad k(\mathbf{x}_i,\mathbf{x}_j)=\phi(\mathbf{x}_i)^{\top}\phi(\mathbf{x}_j)$'});
            plotInput(left,X,y,linear,false,false);
            imagesc(right,X.'*X); axis(right,'square'); colormap(right,parula); colorbar(right);
            xlabel(right,'Observation j'); ylabel(right,'Observation i');
            title(right,'Linear Gram matrix: X^T X'); set(right,'FontSize',12);
            set(notes,'String',{'The data enter the dual through pairwise inner products.', ...
                'Replace these values with a kernel matrix, then solve again for alpha and beta_0.', ...
                'The labels and hard-margin constraints stay the same. There is no box bound C.'});
        else
            formula({dual,prediction});
            plotInput(left,X,y,model,true,stage>=4);
            if stage==3
                imagesc(right,model.K); axis(right,'square'); colormap(right,parula); colorbar(right);
                xlabel(right,'Observation j'); ylabel(right,'Observation i');
                title(right,kernelTitle(kind,sigma),'Interpreter','none'); set(right,'FontSize',12);
            elseif stage==4
                drawGeometry();
            else
                if model.ok
                    stem(right,1:numel(y),model.alpha,'Marker','none','Color',[.3 .3 .35]);
                    hold(right,'on');
                    scatter(right,find(model.sv),model.alpha(model.sv),45,[.12 .12 .12],'filled');
                    xlabel(right,'Observation i'); ylabel(right,'\alpha_i');
                    title(right,'Nonzero coefficients identify support vectors'); grid(right,'on');
                    set(right,'FontSize',12);
                else
                    axis(right,'off'); text(right,.03,.7,model.message,'Units','normalized','FontSize',14);
                end
            end
            if ~model.ok
                lines={model.message,'A different kernel may make these data separable.', ...
                    'Numerical failure is reported separately from an infeasible model.'};
            elseif stage==3
                lines={kernelDescription(kind),summary(model), ...
                    'Solid green: d(x) = 0. Dashed: d(x) = +/-1. Shading shows predicted classes.'};
            elseif stage==4 && kind==2
                lines={'The right plot shows the exact three-dimensional feature map.', ...
                    'The fitted plane and its two parallel margin planes become curves in input space.', ...
                    'Black rings mark support vectors. The animation is only a visual interpolation.'};
            elseif stage==4 && kind==3
                lines={'RBF has an infinite-dimensional feature map; it is not displayed here.', ...
                    'The right plot is the score d(x) over the INPUT plane, with the zero-height plane.', ...
                    'Change sigma to see how the fitted boundary changes. Perfect training fit does not imply good prediction.'};
            elseif stage==4
                lines={'For the linear kernel, phi(x) = x, so feature space is the input space.',summary(model)};
            else
                lines={sprintf('Black rings: %d support vectors (positive alpha above numerical tolerance).',sum(model.sv)), ...
                    'KKT condition: alpha_i [y_i d(x_i) - 1] = 0. Support vectors lie on the margin.', ...
                    sprintf('Minimum y_i d(x_i): %.6f. Feature-space margin width 2/||beta||: %.4f.',min(model.margin),2/sqrt(model.norm2))};
            end
            set(notes,'String',lines);
        end
        drawnow;
    end
    function drawGeometry()
        if ~model.ok
            axis(right,'off'); text(right,.02,.8,model.message,'Units','normalized','FontSize',14); return
        end
        if kind==2
            Z=featureMap(X,2); drawPoints3(right,Z,y,model.sv);
            beta=Z*(model.alpha.*y);
            plane(right,Z,beta,model.beta0,0,green,.23);
            plane(right,Z,beta,model.beta0,-1,blue,.07);
            plane(right,Z,beta,model.beta0,1,red,.07);
            featureAxes();
        elseif kind==1
            plotInput(right,X,y,model,true,true); title(right,'Feature space equals input space');
        else
            limits=inputLimits(X); [u,v]=meshgrid(linspace(limits(1),limits(2),65),linspace(limits(3),limits(4),65));
            G=[u(:).';v(:).']; d=score(model,X,y,G);
            surf(right,u,v,reshape(d,size(u)),'EdgeColor','none','FaceAlpha',.80);
            hold(right,'on');
            surf(right,u([1 end],[1 end]),v([1 end],[1 end]),zeros(2), ...
                'FaceColor',green,'FaceAlpha',.22,'EdgeColor',green);
            drawPoints3(right,[X;model.trainScore.'],y,model.sv);
            xlabel(right,'x_1'); ylabel(right,'x_2'); zlabel(right,'d(x)');
            title(right,'Decision score, not RBF feature space');
            colormap(right,parula); view(right,[-42 25]); grid(right,'on');
        end
    end
    function featureAxes()
        xlabel(right,'z_1 = x_1^2'); ylabel(right,'z_2 = x_2^2');
        zlabel(right,'z_3 = \surd2 x_1 x_2');
        title(right,'Exact quadratic feature space');
        view(right,[-48 24]); grid(right,'on'); set(right,'FontSize',11); axis(right,'vis3d');
    end
    function animate()
        if busy || kind~=2 || stage~=4 || ~model.ok, return; end
        busy=true; set(controls,'Enable','off');
        cla(right,'reset'); P=[X;zeros(1,size(X,2))]; Z=featureMap(X,2);
        h=drawPoints3(right,P,y,false(size(y))); featureAxes();
        R=[P Z]; lo=min(R,[],2); hi=max(R,[],2); pad=.10*max(hi-lo,.2);
        xlim(right,[lo(1)-pad(1) hi(1)+pad(1)]); ylim(right,[lo(2)-pad(2) hi(2)+pad(2)]);
        zlim(right,[lo(3)-pad(3) hi(3)+pad(3)]);
        title(right,'Mapping the same labelled observations');
        for t=linspace(0,1,60)
            if ~isgraphics(fig), return; end
            s=t*t*(3-2*t); Q=(1-s)*P+s*Z;
            for j=1:2
                if j==1, mask=y<0; else, mask=y>0; end
                set(h(j),'XData',Q(1,mask),'YData',Q(2,mask),'ZData',Q(3,mask));
            end
            drawnow; pause(.02);
        end
        busy=false; render(); enableControls();
    end
    function compareKernels()
        if busy, return; end
        busy=true; set(controls,'Enable','off'); drawnow;
        try
            f=figure('Name','Same data, different kernels','NumberTitle','off','Color','w', ...
                'Units','normalized','Position',[.04 .15 .92 .68]);
            for k=1:3
                ax=subplot(1,3,k,'Parent',f); m=fitted(k);
                plotInput(ax,X,y,m,true,true);
                title(ax,{kernelTitle(k,sigma),shortStatus(m)},'Interpreter','none','FontSize',12);
            end
            annotation(f,'textbox',[.06 .01 .88 .075], ...
                'String','Same data and labels. Each kernel is fitted separately. Black rings: support vectors. Margin widths belong to different feature spaces and are not directly comparable.', ...
                'EdgeColor','none','FontSize',12,'HorizontalAlignment','center');
        catch err
            warning('svmDemo:Comparison','%s',err.message);
        end
        busy=false; enableControls();
    end
end

function [X,y]=dataSet(id)
% Fixed deterministic clouds: changing kernels never changes the data.
if id==1
    j=0:31; t=2*pi*j/32; r=.18+.58*(.5+.5*sin(7*t+.4));
    U=[r.*cos(t);r.*sin(t)];
    j=0:39; t=2*pi*j/40; r=1.25+.20*(.5+.5*cos(5*t+.3));
    V=[r.*cos(t);r.*sin(t)]; X=[U V]; y=[-ones(32,1);ones(40,1)];
elseif id==2
    t=linspace(.04,pi-.04,34); r=1+.045*sin(11*t);
    U=[r.*cos(t);r.*sin(t)]; V=[1-r.*cos(t);.42-r.*sin(t)];
    X=[U V]; y=[-ones(34,1);ones(34,1)];
else
    j=1:30; t=j*pi*(3-sqrt(5)); r=.45*sqrt(j/30);
    U=[r.*cos(t);r.*sin(t)];
    V=bsxfun(@plus,U,[1.1;.25]); U=bsxfun(@plus,U,[-1.1;-.25]);
    X=[U V]; y=[-ones(30,1);ones(30,1)];
end
end

function Z=featureMap(X,kind)
if kind==1, Z=X;
elseif kind==2, Z=[X(1,:).^2;X(2,:).^2;sqrt(2)*X(1,:).*X(2,:)];
else, Z=[];
end
end

function K=kernel(A,B,kind,sigma)
if kind==1, K=A.'*B;
elseif kind==2, K=(A.'*B).^2;
else
    D=bsxfun(@plus,sum(A.^2,1).',sum(B.^2,1))-2*(A.'*B);
    K=exp(-max(D,0)/(2*sigma^2));
end
end

function m=fitHardMargin(X,y,kind,sigma)
n=numel(y); K=kernel(X,X,kind,sigma); K=(K+K.')/2;
m=struct('ok',false,'status','numerical','message','Numerical solve not validated.', ...
    'K',K,'kind',kind,'sigma',sigma,'alpha',[],'beta0',[],'sv',false(n,1));
if kind<=2
    Z=featureMap(X,kind);
    % An independent feasibility check avoids treating an unbounded dual
    % iteration or a solver timeout as a fitted hard-margin classifier.
    B=-bsxfun(@times,y,[Z.' ones(n,1)]);
    lp=optimoptions('linprog','Display','none');
    [theta,~,flag]=linprog(zeros(size(B,2),1),B,-ones(n,1),[],[],[],[],lp);
    if flag==-2
        m.status='infeasible'; m.message='No hard-margin separator exists for this kernel.'; return
    elseif flag<=0 || isempty(theta) || max(B*theta+1)>1e-5
        m.message='Feasibility check was numerically inconclusive. No classifier is shown.'; return
    end
end
H=(y*y.').*K; H=(H+H.')/2;
opts=optimoptions('quadprog','Algorithm','interior-point-convex','Display','none', ...
    'OptimalityTolerance',1e-9,'ConstraintTolerance',1e-9,'MaxIterations',1000);
% ub=[] is intentional: this is the hard-margin dual, not large-C soft margin.
[alpha,~,flag]=quadprog(H,-ones(n,1),[],[],y.',0,zeros(n,1),[],[],opts);
if flag<=0 || isempty(alpha) || any(~isfinite(alpha))
    m.message=sprintf('QP solver did not converge (exit flag %d). No classifier is shown.',flag); return
end
tol=1e-7*max(1,max(alpha)); sv=alpha>tol;
if ~any(sv & y>0) || ~any(sv & y<0)
    m.message='Could not identify reliable support vectors from both classes.'; return
end
g=K*(alpha.*y); beta0=mean(y(sv)-g(sv)); d=g+beta0; margin=y.*d;
norm2=alpha.'*H*alpha; dual=sum(alpha)-.5*norm2;
eq=abs(y.'*alpha); complement=max(abs(alpha.*(margin-1)));
gap=abs(norm2-sum(alpha))/max(1,abs(dual));
if min(margin)<1-2e-4 || eq>1e-6*max(1,sum(alpha)) || ...
        min(alpha)<-1e-8 || complement>2e-4*max(1,max(alpha)) || gap>2e-4 || norm2<=0
    m.message='Solver result failed the margin/KKT checks. Try a different kernel or RBF width.'; return
end
m.ok=true; m.status='solved'; m.message='Hard-margin solution verified.';
m.alpha=alpha; m.beta0=beta0; m.sv=sv; m.trainScore=d; m.margin=margin;
m.norm2=norm2; m.eqResidual=eq; m.complementarity=complement; m.relativeGap=gap;
if kind<=2
    m.beta=featureMap(X,kind)*(alpha.*y);
    m.mapError=max(abs(m.beta.'*featureMap(X,kind)+beta0-d.'));
end
end

function d=score(m,X,y,G)
d=(m.alpha.*y).'*kernel(X,G,m.kind,m.sigma)+m.beta0;
end
function limits=inputLimits(X)
lo=min(X,[],2); hi=max(X,[],2); c=(lo+hi)/2; span=max(hi-lo)*.62;
limits=[c(1)-span c(1)+span c(2)-span c(2)+span];
end
function plotInput(ax,X,y,m,showFit,showSV)
blue=[.10 .40 .82]; red=[.86 .22 .20]; green=[.10 .53 .32];
hold(ax,'on'); limits=inputLimits(X);
if showFit && m.ok
    [u,v]=meshgrid(linspace(limits(1),limits(2),160),linspace(limits(3),limits(4),160));
    d=reshape(score(m,X,y,[u(:).';v(:).']),size(u));
    contourf(ax,u,v,double(d>0),[-.5 .5 1.5],'LineStyle','none');
    colormap(ax,[.91 .95 1;1 .94 .93]); caxis(ax,[0 1]);
    contour(ax,u,v,d,[0 0],'Color',green,'LineWidth',2.2);
    contour(ax,u,v,d,[-1 -1],'--','Color',blue,'LineWidth',1.2);
    contour(ax,u,v,d,[1 1],'--','Color',red,'LineWidth',1.2);
elseif showFit && strcmp(m.status,'infeasible') && m.kind==1
    for label=[-1 1]
        U=X(:,y==label); h=convhull(U(1,:),U(2,:));
        if label<0, c=blue; else, c=red; end
        patch(ax,U(1,h),U(2,h),c,'FaceAlpha',.12,'EdgeColor',c);
    end
end
h1=scatter(ax,X(1,y<0),X(2,y<0),40,blue,'filled');
h2=scatter(ax,X(1,y>0),X(2,y>0),48,red,'x','LineWidth',1.5);
if showSV && m.ok
    scatter(ax,X(1,m.sv),X(2,m.sv),115,'k','o','LineWidth',1.2);
end
axis(ax,'equal'); axis(ax,limits); box(ax,'on'); grid(ax,'on');
xlabel(ax,'x_1'); ylabel(ax,'x_2'); set(ax,'FontSize',12);
title(ax,'Input space'); legend(ax,[h1 h2],{'y = -1','y = +1'}, ...
    'Location','northoutside','Orientation','horizontal');
end
function h=drawPoints3(ax,Z,y,sv)
hold(ax,'on');
h(1)=scatter3(ax,Z(1,y<0),Z(2,y<0),Z(3,y<0),40,[.10 .40 .82],'filled');
h(2)=scatter3(ax,Z(1,y>0),Z(2,y>0),Z(3,y>0),48,[.86 .22 .20],'x','LineWidth',1.5);
if any(sv), scatter3(ax,Z(1,sv),Z(2,sv),Z(3,sv),115,'k','o','LineWidth',1.2); end
end
function plane(ax,Z,beta,beta0,level,c,opacity)
% Parameterize the plane using the largest normal component, so vertical
% planes are handled without dividing by a near-zero third component.
[~,j]=max(abs(beta)); other=setdiff(1:3,j);
lo=min(Z,[],2); hi=max(Z,[],2); pad=.10*max(hi-lo,.2); lo=lo-pad; hi=hi+pad;
[u,v]=meshgrid(linspace(lo(other(1)),hi(other(1)),30), ...
    linspace(lo(other(2)),hi(other(2)),30));
P=cell(1,3); P{other(1)}=u; P{other(2)}=v;
P{j}=(level-beta0-beta(other(1))*u-beta(other(2))*v)/beta(j);
P{j}(P{j}<lo(j) | P{j}>hi(j))=NaN;
surf(ax,P{1},P{2},P{3},'FaceColor',c,'FaceAlpha',opacity,'EdgeColor','none');
xlim(ax,[lo(1) hi(1)]); ylim(ax,[lo(2) hi(2)]); zlim(ax,[lo(3) hi(3)]);
end
function t=kernelTitle(k,s)
if k==1, t='Linear: k(x,u) = x^T u';
elseif k==2, t='Quadratic: k(x,u) = (x^T u)^2';
else, t=sprintf('RBF: exp(-||x-u||^2/(2 sigma^2)), sigma = %.2f',s);
end
end
function t=kernelDescription(k)
if k==1, t='Linear kernel: phi(x) = x. The decision boundary is a straight line.';
elseif k==2, t='Quadratic: phi(x) = (x_1^2, x_2^2, sqrt(2) x_1 x_2). A plane gives a quadratic boundary.';
else, t='RBF: nearby points have larger kernel values. Sigma sets the distance scale.';
end
end
function s=summary(m)
s=sprintf('Training errors: %d. Support vectors: %d. Minimum y_i d(x_i): %.6f.', ...
    sum(m.margin<=0),sum(m.sv),min(m.margin));
end
function s=shortStatus(m)
if m.ok, s=sprintf('%d support vectors; zero training errors',sum(m.sv));
elseif strcmp(m.status,'infeasible'), s='Hard-margin constraints infeasible';
else, s='Numerical result not validated'; end
end
function s=onoff(c)
if c, s='on'; else, s='off'; end
end
function results=selftest()
results=struct([]); row=0;
for dataset=1:3
    [X,y]=dataSet(dataset);
    for k=1:3
        if k==3, widths=[.25 .55 1.10]; else, widths=.55; end
        for s=widths
            m=fitHardMargin(X,y,k,s); row=row+1;
            assert(~strcmp(m.status,'numerical'),'%s',m.message);
            if m.ok
                assert(all(m.margin>=1-2e-4));
                if k<=2
                    Z=featureMap(X,k); E=Z.'*Z-m.K;
                    assert(max(abs(E(:)))<1e-10);
                    assert(m.mapError<1e-8);
                end
            end
            if dataset==1 && k==1, assert(strcmp(m.status,'infeasible')); end
            if dataset==1 && k==2, assert(m.ok); end
            if k==3 || (dataset==3 && k==1), assert(m.ok); end
            results(row).dataset=dataset; results(row).kernel=k;
            results(row).sigma=s; results(row).status=m.status;
            fprintf('Data %d, kernel %d, sigma %.2f: %s\n',dataset,k,s,m.status);
        end
    end
end
fprintf('All preset checks passed.\n');
end
