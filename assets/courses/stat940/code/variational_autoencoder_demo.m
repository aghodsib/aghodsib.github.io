function results=variational_autoencoder_demo(mode)
% VARIATIONAL_AUTOENCODER_DEMO  Variational inference and a working VAE.
% Run variational_autoencoder_demo. MATLAB R2016b or later, no toolboxes/data.
% Run variational_autoencoder_demo('selftest') for identities and gradients.
% CLASSROOM ROUTE: use Next to visit the seven scenes. In scene 2 move the
% mean/spread sliders, then Optimize. In scene 3 try starting means of +1
% and -1. In scene 5 change mean/spread before drawing New samples. In
% scene 6 click Train 500 updates (repeat to continue), then click a data
% point to inspect its approximate posterior. Scene 7 samples the model.
% Stop pauses training; Reset restarts the networks in scenes 6 and 7.
% Slides: lec15_from_stat940-allslidespdf_1-15.pdf, printed slides 776-788.
% Retained notation: p(z), p(x|z), p(z|x), q(z), D_KL, L[q(z)].
% The excerpt ends at log p(x)=D_KL[q(z)||p(z|x)]+L[q(z)].
% Encoder, decoder and reparameterization scenes are an explicit continuation.
% q(z) always refers to the selected observation, even in the encoder scenes.
% X is 2-by-n in the VAE. Its prior is N(0,I); its decoder likelihood is
% N(decoder(z),0.2^2 I). Encoder q is diagonal Gaussian. Networks: 2-16-4
% and 2-16-2, tanh hidden layers. Log variances = -2+2*tanh(raw).
% Training uses actual reparameterized gradients and Adam, not a scripted
% motion of points. KL weight is exactly 1. Constants of Gaussian log density
% are included. Evaluation curves are Monte Carlo estimates with fixed noise.
% Decoder(mean of q) is labelled a deterministic reconstruction; it need not
% equal E_q[decoder(z)]. Generation samples BOTH z and observation noise.
% No evidence estimate is presented for the nonlinear neural decoder.
% Reference for continuation: https://arxiv.org/abs/1312.6114
% MATLAB GUI not run in authoring environment. Numerical tests are included.
if nargin<1, mode='gui'; end
if strcmpi(mode,'selftest'), results=selftest(); return; end
if ~strcmpi(mode,'gui'), error('Use ''gui'' or ''selftest''.'); end
results=[]; stream=RandStream('mt19937ar','Seed',940);
scene=1; x=1.5; tau=.6; mu=0; sig=1; busy=false; stopped=false; selected=1;
epsSample=randn(stream,1,60); P=initNet(); M=zeroNet(P); V=M; iter=0; hist=[];
U=randn(stream,2,160); X=[U(1,:);.55*U(2,:)+.45*(U(1,:).^2-1)];
Eeval=randn(stream,2,160); Zgen=randn(stream,2,180); Ogen=randn(stream,2,180);
blue=[.12 .42 .80]; orange=[.88 .38 .10]; green=[.12 .57 .35]; bg=[.97 .98 1];
f=figure('Name','Variational inference and variational autoencoder','NumberTitle','off', ...
    'Color',bg,'Units','normalized','Position',[.025 .06 .95 .85],'CloseRequestFcn',@closeFigure);
head=textControl([.04 .93 .92 .05],'',22);
form=axes('Parent',f,'Position',[.04 .82 .92 .09]);
ax1=axes('Parent',f,'Position',[.065 .39 .25 .36]);
ax2=axes('Parent',f,'Position',[.39 .39 .24 .36]);
ax3=axes('Parent',f,'Position',[.71 .39 .24 .36]);
status=textControl([.05 .285 .90 .055],'',13);
notes=textControl([.05 .17 .90 .10],'',14);
menu=uicontrol(f,'Style','popupmenu','Units','normalized','Position',[.04 .055 .27 .045], ...
    'FontSize',11,'String',{'1. Prior, likelihood, posterior','2. Match q(z): the lower bound', ...
    '3. A posterior with two peaks','4. One encoder for many observations','5. Reparameterize a sample', ...
    '6. Train the VAE','7. Reconstruct and generate'},'Callback',@choose);
back=button([.33 .055 .08 .045],'< Back',@(~,~)advance(-1));
next=button([.42 .055 .08 .045],'Next >',@(~,~)advance(1));
action=button([.52 .055 .16 .045],'Optimize q(z)',@(~,~)act());
resample=button([.70 .055 .13 .045],'New samples',@(~,~)sample());
stop=button([.85 .055 .06 .045],'Stop',@(~,~)halt());
reset=button([.92 .055 .06 .045],'Reset',@(~,~)restart());
sl=gobjects(1,4); lab=gobjects(1,4);
for j=1:4
    xx=.06+(j-1)*.235;
    sl(j)=uicontrol(f,'Style','slider','Units','normalized','Position',[xx .115 .20 .023], ...
        'Callback',@(src,~)adjust(src,j));
    lab(j)=textControl([xx .14 .21 .025],'',10);
end
set(sl(1),'Min',-2,'Max',3,'Value',x); set(sl(2),'Min',.2,'Max',1.3,'Value',tau);
set(sl(3),'Min',-3,'Max',3,'Value',mu); set(sl(4),'Min',.15,'Max',1.8,'Value',sig);
controls=[menu back next action resample reset sl]; render();
    function h=textControl(pos,str,sz)
        h=uicontrol(f,'Style','text','Units','normalized','Position',pos,'String',str, ...
            'FontSize',sz,'BackgroundColor',bg,'HorizontalAlignment','left');
    end
    function h=button(pos,str,cb)
        h=uicontrol(f,'Style','pushbutton','Units','normalized','Position',pos,'String',str,'FontSize',11,'Callback',cb);
    end
    function choose(~,~)
        if busy, return; end
        scene=get(menu,'Value'); presets();
    end
    function advance(d)
        if busy, return; end
        scene=min(7,max(1,scene+d)); presets();
    end
    function presets()
        if scene==3, x=2.5; tau=.35; mu=.6; sig=.7;
        elseif scene<=4, x=1.5; tau=.6; mu=0; sig=1;
        end
        render();
    end
    function restart()
        if busy, return; end
        if scene>=6, P=initNet(); M=zeroNet(P); V=M; iter=0; hist=[];
        elseif scene==3, mu=.6; sig=.7;
        else, mu=0; sig=1; epsSample=randn(stream,1,60); end
        render();
    end
    function adjust(src,j)
        if busy, return; end
        value=get(src,'Value');
        if j==1, x=value; elseif j==2, tau=value; elseif j==3, mu=value; else, sig=value; end
        render();
    end
    function halt(), stopped=true; end
    function closeFigure(~,~), stopped=true; delete(f); end
    function sample()
        if busy, return; end
        epsSample=randn(stream,1,60); Zgen=randn(stream,2,180); Ogen=randn(stream,2,180); render();
    end
    function enabled()
        if ~isgraphics(f), return; end
        set(controls,'Enable',onoff(~busy)); set(stop,'Enable',onoff(busy));
        if ~busy
            set(back,'Enable',onoff(scene>1)); set(next,'Enable',onoff(scene<7));
            set(action,'Enable',onoff(scene==2 || scene==3 || scene>=6));
        end
        set(sl,'Visible',onoff(scene<=5)); set(lab,'Visible',onoff(scene<=5));
        if scene<=5 && ~busy
            set(sl(1:2),'Enable',onoff(scene~=5));
            set(sl(3:4),'Enable',onoff(scene==2 || scene==3 || scene==5));
        end
        if scene>=6, set(action,'String','Train 500 updates'); else, set(action,'String','Optimize q(z)'); end
    end
    function act()
        if busy, return; end
        busy=true; stopped=false; enabled();
        try
            if scene<=3
                u=[mu;log(sig)];
                for j=1:70
                    if stopped || ~isgraphics(f), break; end
                    h=1e-5; grad=zeros(2,1); old=bound(u(1),exp(u(2)),x,tau,scene==3);
                    for k=1:2
                        v=u; v(k)=v(k)+h; w=u; w(k)=w(k)-h;
                        grad(k)=(bound(v(1),exp(v(2)),x,tau,scene==3)-bound(w(1),exp(w(2)),x,tau,scene==3))/(2*h);
                    end
                    rate=.15;
                    for k=1:25
                        v=u+rate*grad;
                        v=max([-3;log(.15)],min([3;log(1.8)],v));
                        if bound(v(1),exp(v(2)),x,tau,scene==3)>=old-1e-12, break; end
                        rate=rate/2;
                    end
                    u=v; mu=u(1); sig=exp(u(2)); render(); pause(.025);
                end
            else
                if isempty(hist), record(); end
                for j=1:500
                    if stopped || ~isgraphics(f), break; end
                    E=randn(stream,2,size(X,2));
                    [~,G]=objective(P,[X X],[E -E]);
                    iter=iter+1; [P,M,V]=adam(P,G,M,V,iter);
                    if mod(iter,20)==0, record(); render(); drawnow; end
                end
            end
        catch err
            if isgraphics(f), set(notes,'String',err.message); end
            warning('vaeDemo:Operation','%s',err.message);
        end
        busy=false; if isgraphics(f), render(); end
    end
    function record()
        [nll,~,r,k]=objective(P,[X X],[Eeval -Eeval]);
        hist(end+1,:)=[iter -nll r k];
    end
    function selectPoint(~,~)
        if busy || scene<6, return; end
        pt=get(ax1,'CurrentPoint'); [~,selected]=min(sum((X-pt(1,1:2).').^2,1)); render();
    end
    function equation(s)
        cla(form); axis(form,'off');
        sz=18; if iscell(s), sz=16; end
        text(form,0,.65,s,'Interpreter','latex','FontSize',sz,'Units','normalized');
    end
    function render()
        if ~isgraphics(f), return; end
        for aa=[ax1 ax2 ax3], cla(aa,'reset'); hold(aa,'on'); grid(aa,'on'); set(aa,'FontSize',11); end
        set(menu,'Value',scene); set(sl(1),'Value',x); set(sl(2),'Value',tau); set(sl(3),'Value',mu); set(sl(4),'Value',sig);
        set(lab(1),'String',sprintf('Observed x = %.2f',x)); set(lab(2),'String',sprintf('Observation noise = %.2f',tau));
        set(lab(3),'String',sprintf('q(z) mean = %.2f',mu)); set(lab(4),'String',sprintf('q(z) standard deviation = %.2f',sig));
        names={'Use the observation to infer z','Move q(z) toward the posterior','One Gaussian cannot match two peaks', ...
            'Predict the parameters of q(z)','Keep the noise; move the sample','Learn the encoder and decoder together','Reconstruction and generation'};
        set(head,'String',sprintf('%d / 7    %s',scene,names{scene}));
        set(status,'String','');
        z=linspace(-5,5,1600);
        if scene<=3
            squared=scene==3; [evidence,post]=posterior(z,x,tau,squared);
            if squared, meanx=z.^2; else, meanx=z; end
            likelihood=normal(x,meanx,tau); prior=normal(z,0,1); approx=normal(z,mu,sig);
            L=bound(mu,sig,x,tau,squared); gap=log(evidence)-L;
            if scene==1
                equation('$p(z\mid x)=\frac{p(x\mid z)p(z)}{p(x)},\qquad p(x)=\int p(x\mid z)p(z)\,dz$');
                plot(ax1,z,prior,'Color',[.45 .45 .45],'LineWidth',2); title(ax1,'Prior p(z)');
                plot(ax2,z,likelihood,'Color',orange,'LineWidth',2); title(ax2,{'Likelihood p(x | z)','as a function of z'});
                plot(ax3,z,post,'Color',blue,'LineWidth',2); title(ax3,'Posterior p(z | x)');
                lines={'Choose an observed x. Which latent z values explain it?', ...
                    'The posterior combines the likelihood with the prior. More noise gives less certainty.'};
            else
                equation({'$\log p(x)=D_{KL}[q(z)\Vert p(z\mid x)]+L[q(z)]$', ...
                    '$L[q(z)]=E_{q(z)}[\log p(z,x)]-E_{q(z)}[\log q(z)]$'});
                plot(ax1,z,post,'Color',blue,'LineWidth',2); plot(ax1,z,approx,'Color',orange,'LineWidth',2);
                title(ax1,'Target and approximation'); legend(ax1,{'p(z | x)','q(z)'},'Location','northwest');
                plot(ax2,[0 1],[log(evidence) log(evidence)],'Color',blue,'LineWidth',2);
                plot(ax2,[0 1],[L L],'Color',orange,'LineWidth',2);
                plot(ax2,[.55 .55],[L log(evidence)],'Color',green,'LineWidth',5);
                text(ax2,.58,(L+log(evidence))/2,sprintf('KL gap\n%.3f',gap),'FontSize',12);
                xlim(ax2,[0 1]); set(ax2,'XTick',[]); title(ax2,{'Evidence stays fixed','while q(z) changes'});
                legend(ax2,{'log p(x)','L[q(z)]'},'Location','southwest');
                [ll,kl]=terms(mu,sig,x,tau,squared);
                bar(ax3,[ll -kl L]); set(ax3,'XTick',1:3,'XTickLabel',{'E_q log p(x|z)','-KL(q || prior)','L[q(z)]'}); xtickangle(ax3,18);
                title(ax3,'Contributions to L[q(z)]');
                if squared
                    lines={'This example uses p(x | z) = N(z^2, noise^2). Positive and negative z can explain x.', ...
                        'Try a positive or negative starting mean, then Optimize. The Gaussian family limits the fit.'};
                else
                    lines={'Move the mean and spread of q(z), or click Optimize q(z).', ...
                        'As q approaches the posterior, the KL gap closes and the lower bound rises.'};
                end
                set(status,'String',sprintf('log p(x) = %.4f     L[q(z)] = %.4f     D_KL[q(z) || p(z|x)] = %.4f',log(evidence),L,gap));
            end
            for aa=[ax1 ax2 ax3], if scene==1 || aa==ax1, xlabel(aa,'z'); ylabel(aa,'Density / likelihood'); xlim(aa,[-5 5]); end; end
        elseif scene==4
            equation('$x\longmapsto (\mu(x),\sigma(x))\longmapsto q(z),\qquad q(z)\approx p(z\mid x)$');
            xs=linspace(-3,3,80); means=xs/(1+tau^2); sd=tau/sqrt(1+tau^2);
            plot(ax1,xs,means,'Color',orange,'LineWidth',2); scatter(ax1,x,x/(1+tau^2),70,blue,'filled');
            xlabel(ax1,'Observed x'); ylabel(ax1,'Mean of q(z)'); title(ax1,'One shared mean rule');
            plot(ax2,xs,sd*ones(size(xs)),'Color',orange,'LineWidth',2); ylim(ax2,[0 1.2]);
            xlabel(ax2,'Observed x'); ylabel(ax2,'Standard deviation'); title(ax2,'One shared uncertainty rule');
            plot(ax3,z,normal(z,x/(1+tau^2),sd),'Color',orange,'LineWidth',2); xlabel(ax3,'z'); title(ax3,'q(z) for the selected x');
            lines={'Here a shared rule gives the exact Gaussian posterior for every x. It is calculated analytically.', ...
                'A VAE uses a neural encoder to learn such a rule. Its q(z) is still an approximate posterior.'};
        elseif scene==5
            equation('$\epsilon\sim N(0,1),\qquad z=\mu+\sigma\epsilon,\qquad z\sim q(z)$');
            plot(ax1,z,normal(z,0,1),'Color',[.4 .4 .4],'LineWidth',2);
            scatter(ax1,epsSample,zeros(size(epsSample)),20,blue,'filled'); title(ax1,'Fixed noise samples'); xlabel(ax1,'epsilon');
            plot(ax2,[-3 3],mu+sig*[-3 3],'Color',orange,'LineWidth',2);
            scatter(ax2,epsSample,mu+sig*epsSample,20,blue,'filled'); xlabel(ax2,'epsilon'); ylabel(ax2,'z'); title(ax2,'Shift and scale');
            plot(ax3,z,normal(z,mu,sig),'Color',orange,'LineWidth',2); scatter(ax3,mu+sig*epsSample,zeros(size(epsSample)),20,blue,'filled');
            title(ax3,'Samples from q(z)'); xlabel(ax3,'z');
            lines={'The noise points stay fixed while you change the mean or standard deviation.', ...
                'Each sampled z moves smoothly. Click New samples to draw fresh noise.'};
        else
            [Mu,S]=encode(P,X); recon=decode(P,Mu); z0=Mu(:,selected); s0=S(:,selected);
            equation('$L[q(z)]=E_{q(z)}[\log p(x\mid z)]-D_{KL}[q(z)\Vert p(z)]$');
            dots=scatter(ax1,X(1,:),X(2,:),18,blue,'filled'); set(dots,'ButtonDownFcn',@selectPoint); set(ax1,'ButtonDownFcn',@selectPoint);
            scatter(ax1,recon(1,:),recon(2,:),16,orange,'x');
            plot(ax1,[X(1,selected) recon(1,selected)],[X(2,selected) recon(2,selected)],'k-','LineWidth',2);
            scatter(ax1,X(1,selected),X(2,selected),95,'k','o');
            title(ax1,'Data and decoder(mean of q)'); legend(ax1,{'Data','Deterministic reconstruction'},'Location','northwest','FontSize',9);
            xlabel(ax1,'x_1'); ylabel(ax1,'x_2'); axis(ax1,'equal');
            scatter(ax2,Mu(1,:),Mu(2,:),15,blue,'filled');
            angle=linspace(0,2*pi,100); ellipse=z0+2*s0.*[cos(angle);sin(angle)];
            plot(ax2,ellipse(1,:),ellipse(2,:),'Color',orange,'LineWidth',2);
            draw=bsxfun(@plus,z0,bsxfun(@times,s0,[epsSample(1:30);epsSample(31:60)]));
            scatter(ax2,draw(1,:),draw(2,:),22,orange,'filled'); plot(ax2,2*cos(angle),2*sin(angle),'k:');
            title(ax2,'Encoder means and selected q(z)'); xlabel(ax2,'z_1'); ylabel(ax2,'z_2'); axis(ax2,'equal');
            if scene==6
                if isempty(hist), record(); end
                plot(ax3,hist(:,1),hist(:,2),'.-','Color',green,'LineWidth',2);
                plot(ax3,hist(:,1),hist(:,3),'Color',orange,'LineWidth',1.2);
                plot(ax3,hist(:,1),-hist(:,4),'Color',blue,'LineWidth',1.2);
                xlabel(ax3,'Training update'); title(ax3,'Estimated average lower bound');
                legend(ax3,{'L[q(z)]','E_q log p(x|z)','-KL(q || prior)'},'Location','best');
                lines={'Click Train 500 updates. Both networks learn from the same lower-bound objective.', ...
                    'Click a data point to inspect its q(z). The orange ellipse shows two standard deviations.'};
            else
                centers=decode(P,Zgen); generated=centers+.2*Ogen;
                scatter(ax3,X(1,:),X(2,:),12,[.75 .75 .75],'filled'); scatter(ax3,generated(1,:),generated(2,:),20,green,'filled');
                xlabel(ax3,'x_1'); ylabel(ax3,'x_2'); axis(ax3,'equal'); title(ax3,'New samples: prior, then decoder');
                legend(ax3,{'Training data','Generated observations'},'Location','northwest');
                lines={'Reconstruction starts from an observed x and its q(z). Generation starts from z drawn from p(z).', ...
                    'The decoder gives a Gaussian distribution. Generated observations include its noise, not only its mean.'};
            end
            set(status,'String',sprintf('Training updates: %d. Selected observation: %d. Decoder noise SD: 0.20. Dotted circle: prior 2-SD contour.',iter,selected));
        end
        set(notes,'String',lines); enabled(); drawnow;
    end
end
function p=normal(x,m,s)
p=exp(-.5*((x-m)./s).^2)./(sqrt(2*pi)*s);
end
function [e,p]=posterior(z,x,tau,squared)
if squared
    e=integral(@(u)normal(x,u.^2,tau).*normal(u,0,1),-Inf,Inf,'AbsTol',1e-12);
    p=normal(x,z.^2,tau).*normal(z,0,1)/e;
else
    e=normal(x,0,sqrt(1+tau^2)); p=normal(z,x/(1+tau^2),tau/sqrt(1+tau^2));
end
end
function [ll,kl]=terms(mu,s,x,tau,squared)
if squared, residual=x*x-2*x*(mu^2+s^2)+mu^4+6*mu^2*s^2+3*s^4;
else, residual=(x-mu)^2+s^2; end
ll=-.5*log(2*pi*tau^2)-residual/(2*tau^2);
kl=.5*(mu^2+s^2-1-log(s^2));
end
function L=bound(mu,s,x,tau,squared)
[ll,kl]=terms(mu,s,x,tau,squared); L=ll-kl;
end
function P=initNet()
P.A=reshape(.4*sin(1:32),16,2); P.b=zeros(16,1);
P.B=reshape(.16*cos(1:64),4,16); P.c=zeros(4,1);
P.C=reshape(.4*cos(1:32),16,2); P.d=zeros(16,1);
P.D=reshape(.16*sin(1:32),2,16); P.e=zeros(2,1);
end
function Q=zeroNet(P)
Q=P; keys=fieldnames(P); for j=1:numel(keys), Q.(keys{j})=zeros(size(P.(keys{j}))); end
end
function [mu,s,h,r]=encode(P,X)
h=tanh(P.A*X+P.b); r=P.B*h+P.c; mu=r(1:2,:); s=exp(-1+tanh(r(3:4,:)));
end
function [out,h]=decode(P,Z)
h=tanh(P.C*Z+P.d); out=P.D*h+P.e;
end
function [cost,G,ll,kl]=objective(P,X,E)
n=size(X,2); [mu,s,h,r]=encode(P,X); Z=mu+s.*E; [out,h2]=decode(P,Z);
R=out-X; variance=.04;
ll=-mean(sum(R.^2,1))/(2*variance)-log(2*pi*variance);
kl=.5*mean(sum(mu.^2+s.^2-1-2*log(s),1)); cost=-ll+kl;
if nargout<2, return; end
v=R/(variance*n); G.D=v*h2.'; G.e=sum(v,2);
v=(P.D.'*v).*(1-h2.^2); G.C=v*Z.'; G.d=sum(v,2); dz=P.C.'*v;
dmu=dz+mu/n; dlogS=dz.*s.*E+(s.^2-1)/n;
v=[dmu;dlogS.*(1-tanh(r(3:4,:)).^2)];
G.B=v*h.'; G.c=sum(v,2); v=(P.B.'*v).*(1-h.^2);
G.A=v*X.'; G.b=sum(v,2);
end
function [P,M,V]=adam(P,G,M,V,t)
keys=fieldnames(P);
for j=1:numel(keys)
    k=keys{j}; M.(k)=.9*M.(k)+.1*G.(k); V.(k)=.999*V.(k)+.001*G.(k).^2;
    P.(k)=P.(k)-.003*(M.(k)/(1-.9^t))./(sqrt(V.(k)/(1-.999^t))+1e-8);
end
end
function s=onoff(v)
if v, s='on'; else, s='off'; end
end
function results=selftest()
for x=[-1 0 2]
    for tau=[.2 .6 1.2]
        mu=x/(1+tau^2); s=tau/sqrt(1+tau^2);
        [e,~]=posterior(0,x,tau,false); assert(abs(bound(mu,s,x,tau,false)-log(e))<1e-10);
    end
end
P=initNet(); X=[.2 -.4 .7;.3 .5 -.6]; E=[.4 -.8 .1;-.3 .6 .9]; [~,G]=objective(P,X,E);
keys=fieldnames(P); maxError=0; h=1e-5;
for j=1:numel(keys)
    k=keys{j};
    for i=1:numel(P.(k))
        Q=P; R=P; Q.(k)(i)=Q.(k)(i)+h; R.(k)(i)=R.(k)(i)-h;
        numeric=(objective(Q,X,E)-objective(R,X,E))/(2*h);
        maxError=max(maxError,abs(numeric-G.(k)(i)));
    end
end
assert(maxError<1e-5); results=struct('passed',true,'maximumGradientError',maxError);
fprintf('Gaussian identity and all neural gradients passed. Maximum error %.3g\n',maxError);
end
