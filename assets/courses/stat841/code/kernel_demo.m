function kernel_demo
% KERNEL_DEMO  Two visual examples of a quadratic feature map and kernel.
% Save as kernel_demo.m, then run: kernel_demo
% Uses base MATLAB only. No data files or toolboxes are required.
%
% Controls: Next / Back, Replay mapping, example menu, and line-angle slider.
% Drag the right-hand plot to rotate it with MATLAB's Rotate 3D tool.
% Data convention: X is 2-by-n; each column is one observation.
% Phi(X) = [X(1,:).^2; X(2,:).^2; sqrt(2)*X(1,:).*X(2,:)].
%
% The boundaries are constructed, not estimated with an SVM. This isolates
% feature mapping from fitting. The animation between the two spaces is a
% visual interpolation, not a family of kernel feature maps.
% Example 2 represents centered sensor readings and an interaction rule;
% its labels are synthetic, not a physical claim about real sensors.

oldRng = rng; restoreRng = onCleanup(@() rng(oldRng)); %#ok<NASGU>
rng(841);
blue = [0.10 0.40 0.82]; red = [0.86 0.22 0.20];
green = [0.17 0.58 0.39]; bg = [0.97 0.98 1];
phi = @(X) [X(1,:).^2; X(2,:).^2; sqrt(2)*X(1,:).*X(2,:)];
A = [1 0 1; 0 1 1]; % Three anchor vectors (columns).
example = 1; stage = 1; busy = false;
X = []; Z = []; y = []; w = []; b = []; alpha = [];

fig = figure('Name','Kernel demo','NumberTitle','off','Color',bg, ...
    'Units','normalized','Position',[.04 .08 .92 .82]);
heading = uicontrol(fig,'Style','text','Units','normalized', ...
    'Position',[.04 .92 .92 .055],'BackgroundColor',bg, ...
    'FontSize',22,'FontWeight','bold','HorizontalAlignment','left');
formula = uicontrol(fig,'Style','text','Units','normalized', ...
    'Position',[.04 .865 .92 .045],'BackgroundColor',bg, ...
    'FontSize',15,'HorizontalAlignment','left', ...
    'String','phi(x) = (x_1^2, x_2^2, sqrt(2) x_1 x_2)     |     k(x,u) = (x^T u)^2');
ax1 = axes('Parent',fig,'Position',[.07 .34 .37 .47]);
ax2 = axes('Parent',fig,'Position',[.56 .34 .37 .47]);
note = uicontrol(fig,'Style','text','Units','normalized', ...
    'Position',[.055 .115 .89 .15],'BackgroundColor',bg, ...
    'FontSize',15,'HorizontalAlignment','left');
menu = uicontrol(fig,'Style','popupmenu','Units','normalized', ...
    'Position',[.04 .035 .30 .05],'FontSize',12, ...
    'String',{'1. Inner cloud and outer ring','2. Two sensors: interaction matters'}, ...
    'Callback',@changeExample);
back = uicontrol(fig,'Style','pushbutton','Units','normalized', ...
    'Position',[.39 .035 .12 .05],'String','< Back','FontSize',12, ...
    'Callback',@(~,~) step(-1));
next = uicontrol(fig,'Style','pushbutton','Units','normalized', ...
    'Position',[.53 .035 .12 .05],'String','Next >','FontSize',12, ...
    'Callback',@(~,~) step(1));
replay = uicontrol(fig,'Style','pushbutton','Units','normalized', ...
    'Position',[.68 .035 .18 .05],'String','Replay mapping','FontSize',12, ...
    'Callback',@(~,~) animateMap());
angle = uicontrol(fig,'Style','slider','Units','normalized', ...
    'Position',[.12 .285 .22 .025],'Min',0,'Max',180,'Value',30, ...
    'Callback',@(~,~) render());
angleLabel = uicontrol(fig,'Style','text','Units','normalized', ...
    'Position',[.07 .258 .37 .025],'BackgroundColor',bg, ...
    'String','Rotate a candidate straight line','FontSize',11);
makeData(); render();

    function makeData()
        if example == 1
            % Antipodal pairs make the nonseparability exact: the convex
            % hull of each class contains the origin.
            theta = 2*pi*rand(1,70);
            radius = .72*sqrt(rand(1,70));
            U = [radius.*cos(theta); radius.*sin(theta)];
            theta = 2*pi*rand(1,85);
            radius = 1.24 + .38*rand(1,85);
            V = [radius.*cos(theta); radius.*sin(theta)];
            X = [U -U V -V];
            y = [-ones(1,140) ones(1,170)];
            w = [1;1;0]; b = -1; alpha = [1;1;0];
        else
            % Opposite readings map to the same feature vector. Their labels
            % agree because the interaction is unchanged by a sign reversal.
            U = 3.2*rand(2,1400)-1.6;
            score = U(1,:).*U(2,:) - .35;
            U = U(:,abs(score)>.14); % Clear gap around the true boundary.
            score = U(1,:).*U(2,:) - .35;
            V = U(:,find(score<0,75,'first'));
            W = U(:,find(score>0,75,'first'));
            X = [V -V W -W];
            y = [-ones(1,150) ones(1,150)];
            w = [0;0;1/sqrt(2)]; b = -.35;
            alpha = [-.5;-.5;.5];
        end
        Z = phi(X);
        explicit = w.'*Z+b;
        implicit = alpha.'*((A.'*X).^2)+b;
        assert(all(y.*explicit>0),'Constructed boundary failed.');
        assert(max(abs(explicit-implicit))<1e-11,'Kernel scores disagree.');
        assert(max(max(abs(Z.'*Z-(X.'*X).^2)))<1e-10, ...
            'Kernel identity failed.');
    end

    function changeExample(~,~)
        if busy, return; end
        example = get(menu,'Value'); stage = 1;
        makeData(); render();
    end

    function step(delta)
        if busy, return; end
        stage = min(5,max(1,stage+delta)); render();
        if stage==2 && delta>0, animateMap(); end
    end

    function styleAxes(ax)
        set(ax,'FontSize',12,'LineWidth',1,'Box','on','Color','w');
        grid(ax,'on'); hold(ax,'on');
    end

    function h = points2(ax)
        h(1) = scatter(ax,X(1,y<0),X(2,y<0),34,blue,'filled');
        h(2) = scatter(ax,X(1,y>0),X(2,y>0),42,red,'x','LineWidth',1.4);
    end

    function h = points3(P)
        h(1) = scatter3(ax2,P(1,y<0),P(2,y<0),P(3,y<0),34,blue,'filled');
        h(2) = scatter3(ax2,P(1,y>0),P(2,y>0),P(3,y>0),42,red,'x','LineWidth',1.4);
    end

    function render()
        if ~isgraphics(fig), return; end
        cla(ax1,'reset'); cla(ax2,'reset');
        styleAxes(ax1); styleAxes(ax2);
        names = {'Start with the data','Apply the feature map', ...
            'Separate with a plane','Return to the input space','Compute with the kernel'};
        set(heading,'String',sprintf('%d / 5     %s',stage,names{stage}));
        set(back,'Enable',onoff(stage>1)); set(next,'Enable',onoff(stage<5));
        set(replay,'Enable',onoff(stage>=2));
        set(angle,'Visible',onoff(stage==1));
        set(angleLabel,'Visible',onoff(stage==1));
        xlim(ax1,[-1.8 1.8]); ylim(ax1,[-1.8 1.8]); axis(ax1,'square');
        xlabel(ax1,'x_1'); ylabel(ax1,'x_2'); title(ax1,'Input space: two coordinates');
        if stage>=4
            [u,v] = meshgrid(linspace(-1.8,1.8,220));
            if example==1, F=u.^2+v.^2-1; else, F=u.*v-.35; end
            contourf(ax1,u,v,double(F>0),[-.5 .5 1.5],'LineStyle','none');
            colormap(ax1,[.90 .94 1;1 .93 .92]); caxis(ax1,[0 1]);
            contour(ax1,u,v,F,[0 0],'Color',green,'LineWidth',2.5);
        end
        h = points2(ax1);
        if example==1, labels={'Blue: inner cloud','Red: outer ring'};
        else, labels={'Blue: low interaction','Red: high interaction'}; end
        legend(ax1,h,labels,'Location','northoutside','Orientation','horizontal');
        if stage==1
            t = get(angle,'Value')*pi/180;
            plot(ax1,2.5*[-cos(t) cos(t)],2.5*[-sin(t) sin(t)], ...
                '--','Color',[.25 .25 .25],'LineWidth',1.5);
            axis(ax2,'off');
            text(ax2,.05,.70,'Can one straight line separate the colours?', ...
                'Units','normalized','FontSize',16,'FontWeight','bold');
            text(ax2,.05,.42,{'Try rotating the line below the left plot.', ...
                'Moving the line will not solve the problem either:', ...
                'both classes have the origin in their convex hull.'}, ...
                'Units','normalized','FontSize',14);
        else
            points3(Z); setupFeatureAxes(false);
            if stage>=3, drawPlane(); end
        end
        if example==1
            boundary = 'Plane: z_1 + z_2 = 1.     Input boundary: x_1^2 + x_2^2 = 1.';
        else
            boundary = 'Plane: z_3 = sqrt(2) x 0.35.     Input boundary: x_1 x_2 = 0.35.';
        end
        switch stage
            case 1
                lines = {'The two colours cannot be separated by an affine straight line.', ...
                    'The labels and colours will stay fixed as we change coordinates.'};
            case 2
                lines = {'Each point is mapped to (x_1^2, x_2^2, sqrt(2) x_1 x_2).', ...
                    'The animation connects the old and new coordinates; only its endpoint is phi(x).', ...
                    'Opposite points x and -x coincide after mapping. No labels are changed.'};
            case 3
                lines = {boundary, 'The translucent green surface is an affine separating plane.'};
                if example==1
                    lines{3}='Here z_3 is unnecessary for separation. The next example uses it.';
                else
                    lines{3}='The product detects coordinated readings; distance from the origin alone cannot do this.';
                end
            case 4
                lines = {boundary, ...
                    'A plane in feature space becomes a curved boundary in input space.', ...
                    'The shaded regions use exactly the same decision score as the plane.'};
            case 5
                explicit = w.'*Z+b;
                implicit = alpha.'*((A.'*X).^2)+b;
                err = max(abs(explicit-implicit));
                lines = {'phi(x)^T phi(u) = (x_1 u_1 + x_2 u_2)^2 = k(x,u).', ...
                    'Same score: w^T phi(x) + b = sum_j alpha_j k(a_j,x) + b.', ...
                    sprintf('Computed both ways on every point. Maximum difference: %.2g. See Command Window for details.',err)};
                printKernel();
        end
        set(note,'String',lines); drawnow;
    end

    function setupFeatureAxes(animating)
        xlabel(ax2,'z_1 = x_1^2'); ylabel(ax2,'z_2 = x_2^2');
        zlabel(ax2,'z_3 = \surd2 x_1 x_2');
        title(ax2,'Feature space: three coordinates');
        if animating
            xlim(ax2,[-1.8 3]); ylim(ax2,[-1.8 3]);
        else
            xlim(ax2,[0 3]); ylim(ax2,[0 3]);
        end
        zlim(ax2,[-3.9 3.9]); view(ax2,[-47 24]);
        pbaspect(ax2,[1 1 1]);
    end

    function drawPlane()
        if example==1
            [s,t] = meshgrid(linspace(0,1,2),[-3.9 3.9]);
            surf(ax2,s,1-s,t,'FaceColor',green,'FaceAlpha',.24, ...
                'EdgeColor',green,'LineWidth',1);
        else
            [s,t] = meshgrid([0 3],[0 3]);
            surf(ax2,s,t,sqrt(2)*.35*ones(size(s)), ...
                'FaceColor',green,'FaceAlpha',.24,'EdgeColor',green);
        end
    end

    function animateMap()
        if busy || ~isgraphics(fig), return; end
        busy = true;
        set([menu back next replay angle],'Enable','off');
        cla(ax2,'reset'); styleAxes(ax2); setupFeatureAxes(true);
        P = [X; zeros(1,size(X,2))]; h = points3(P);
        title(ax2,'Mapping the same observations...');
        for t = linspace(0,1,65)
            if ~isgraphics(fig), return; end
            s = t*t*(3-2*t); Q = (1-s)*P+s*Z;
            for j = 1:2
                if j==1, mask=y<0; else, mask=y>0; end
                set(h(j),'XData',Q(1,mask),'YData',Q(2,mask),'ZData',Q(3,mask));
            end
            drawnow; pause(.018);
        end
        busy = false;
        if isgraphics(fig)
            set([menu angle],'Enable','on'); render();
        end
    end

    function printKernel()
        fprintf('\n--- Example %d: exact kernel calculation ---\n',example);
        fprintf('Columns of A are anchors a_j:\n'); disp(A);
        fprintf('alpha =\n'); disp(alpha);
        fprintf('w = Phi(A)*alpha =\n'); disp(phi(A)*alpha);
        fprintf('b = %.4f\n',b);
        fprintf('k(x,u) = (x''*u)^2; f(x) = alpha''*(A''*x).^2 + b.\n');
        fprintf('Classify blue if f(x)<0; red if f(x)>0.\n');
        fprintf('These coefficients describe the chosen boundary; they are not fitted.\n');
        fprintf('Kernel learning methods can learn coefficients from labelled data.\n');
        if example==1
            fprintf('f(x) = k([1;0],x) + k([0;1],x) - 1.\n');
        else
            fprintf('f(x) = 0.5*k([1;1],x) - 0.5*k([1;0],x) - 0.5*k([0;1],x) - 0.35.\n');
        end
    end
end

function s = onoff(condition)
if condition, s = 'on'; else, s = 'off'; end
end
