function out=pruning_demo(mode)
% PRUNING_DEMO  Region merging and cost-complexity pruning.
% Run pruning_demo. MATLAB R2016b+, no additional toolboxes or data files.
% Run pruning_demo('selftest') for mathematical checks without a figure.
%
% Reference: w18stat841All-v4.pptx, slides 369-376.
% Retains the slides' Q_m(T), C_m, n_m, R_m, T_0, T_1, ..., s_j, alpha,
% and |T| notation, including uppercase Q and C despite being scalars.
% Q_m(T) = mean squared residual within leaf m.
% Objective = sum_m n_m*Q_m(T) + alpha*|T|; |T| counts LEAVES.
% X is 2-by-n, one observation per column.
%
% The initial tree has five prescribed regions, to illustrate the lecture's
% partition/tree diagram. Constants C_m are fitted to the displayed data.
% The initial topology is illustrative, not a globally optimized tree.
% Weakest-link pruning minimizes SSE increase per removed leaf. Only
% descendants of a common tree node may be merged. Predictions after a
% collapse use the mean of ALL observations in that node (weighted mean).
% The best-tree button minimizes the displayed regularized objective over
% subtrees of this initial tree; it does not select alpha using test data.
%
% Suggested sequence: Preview merge, Merge next, Merge next, Undo.
% Change alpha and compare the three cost curves. Best for alpha selects
% the minimum-cost subtree. Purple marks the proposed/just-completed merge.
if nargin<1, mode='gui'; end
if strcmpi(mode,'selftest'), out=selftest(); return; end
if ~strcmpi(mode,'gui'), error('Use gui or selftest.'); end
[X,y,T0]=example(); [path,removed,thresholds]=prune(T0);
risks=cellfun(@risk,path); sizes=cellfun(@(t)numel(leaves(t)),path);
alpha=2; stage=1; preview=false; last=0;
blue=[.13 .40 .80]; orange=[.90 .40 .10]; green=[.12 .55 .30]; purple=[.60 .15 .72];
f=figure('Name','Pruning: merge regions and simplify the tree','NumberTitle','off', ...
 'Units','normalized','Position',[.02 .04 .96 .89],'Color',[.97 .98 1]); out=f;
label([.035 .95 .93 .035],'Pruning: merge regions and simplify the tree',21);
formula=axes('Parent',f,'Position',[.035 .86 .93 .075]); axis(formula,'off');
text(formula,0,.82,'$Q_m(T)=\frac{1}{n_m}\sum_{\mathbf{x}_i\in R_m}(y_i-C_m)^2,\qquad C_m=\frac{1}{n_m}\sum_{\mathbf{x}_i\in R_m}y_i$', ...
 'Interpreter','latex','FontSize',16);
text(formula,0,.16,'$\mathrm{Total\ cost}=\underbrace{\sum_{m=1}^{|T|}n_m Q_m(T)}_{\mathrm{fitting\ error}}+\underbrace{\alpha |T|}_{\mathrm{regularizer}},\qquad |T|=\mathrm{number\ of\ leaves}$', ...
 'Interpreter','latex','FontSize',16);
a=axes('Parent',f,'Position',[.055 .52 .26 .28]);
b=axes('Parent',f,'Position',[.385 .52 .25 .28]);
c=axes('Parent',f,'Position',[.72 .52 .24 .28]);
label([.04 .425 .53 .035],'Leaf calculations: multiply Q_m(T) by n_m before summing',11);
tab=uitable(f,'Units','normalized','Position',[.04 .205 .49 .215], ...
 'ColumnName',{'Region','n_m','C_m','Q_m(T)','n_m Q_m(T)'}, ...
 'ColumnWidth',{65 65 85 105 115},'RowName',[],'FontSize',11);
d=axes('Parent',f,'Position',[.62 .245 .33 .18]);
status=label([.04 .145 .92 .05],'',12);
notes=label([.04 .087 .92 .053],'',11);
prev=button([.04 .025 .10 .04],'Undo',@(~,~)back());
peek=button([.155 .025 .14 .04],'Preview merge',@(~,~)peekMerge());
next=button([.31 .025 .12 .04],'Merge next',@(~,~)forward());
alphaLabel=label([.46 .06 .14 .023],'alpha (editable):',10);
alphaEdit=uicontrol(f,'Style','edit','Units','normalized','Position',[.60 .055 .08 .03], ...
 'String','2','Callback',@editAlpha);
slider=uicontrol(f,'Style','slider','Units','normalized','Position',[.46 .028 .22 .025], ...
 'Min',0,'Max',1.12*thresholds(end),'Value',alpha,'Callback',@adjust);
button([.71 .025 .14 .04],'Best for alpha',@(~,~)best());
button([.87 .025 .09 .04],'Reset',@(~,~)reset()); render();
 function h=label(pos,str,sz)
  h=uicontrol(f,'Style','text','Units','normalized','Position',pos,'String',str,'FontSize',sz, ...
   'BackgroundColor',[.97 .98 1],'HorizontalAlignment','left');
 end
 function h=button(pos,str,cb)
  h=uicontrol(f,'Style','pushbutton','Units','normalized','Position',pos,'String',str,'FontSize',11,'Callback',cb);
 end
 function peekMerge(), preview=~preview; last=0; render(); end
 function forward()
  if stage<numel(path), last=removed{stage}; stage=stage+1; preview=false; end
  render();
 end
 function back(), stage=max(1,stage-1); preview=false; last=0; render(); end
 function reset(), alpha=2; stage=1; preview=false; last=0; render(); end
 function adjust(src,~), alpha=get(src,'Value'); render(); end
 function editAlpha(src,~)
  value=str2double(get(src,'String'));
  if isfinite(value) && value>=0, alpha=value; end
  render();
 end
 function best()
  totals=risks+alpha*sizes; bestCost=min(totals);
  stage=find(totals<=bestCost+1e-10,1,'last'); preview=false; last=0; render();
 end
 function render()
  delete(findall(f,'Type','colorbar')); delete(findall(f,'Type','legend'));
  set(a,'Position',[.055 .52 .26 .28]); set(d,'Position',[.62 .245 .33 .18]);
  for ax=[a b c d], cla(ax,'reset'); hold(ax,'on'); grid(ax,'on'); set(ax,'FontSize',10); end
  T=path{stage}; ids=leaves(T); marked=last;
  if preview && stage<numel(path), marked=removed{stage}; end
  drawRegions(a,T,X,y,marked,purple); drawTree(b,T,marked,purple); drawSurface(c,T,marked,purple);
  title(a,sprintf('T_%d: %d rectangular regions',stage-1,numel(ids)));
  title(b,'One region per leaf'); title(c,'Height = fitted constant C_m');
  data=cell(numel(ids),5);
  for j=1:numel(ids)
   n=T(ids(j)); data(j,:)={sprintf('R_%d',j),numel(n.idx),n.C,n.SSE/numel(n.idx),n.SSE};
  end
  set(tab,'Data',data); total=risks+alpha*sizes;
  plot(d,0:numel(path)-1,risks,'o-','Color',blue,'LineWidth',1.6);
  plot(d,0:numel(path)-1,alpha*sizes,'o-','Color',orange,'LineWidth',1.6);
  plot(d,0:numel(path)-1,total,'o-','Color',green,'LineWidth',2);
  scatter(d,stage-1,total(stage),100,purple,'LineWidth',2);
  optimal=find(total<=min(total)+1e-10,1,'last'); plot(d,optimal-1,total(optimal),'kp','MarkerSize',12,'MarkerFaceColor',[1 1 .3]);
  set(d,'XTick',0:numel(path)-1,'XTickLabel',arrayfun(@(k)sprintf('T_%d (%d)',k-1,sizes(k)),1:numel(path),'UniformOutput',false));
  xlabel(d,'Pruning sequence (leaf count in parentheses)'); ylabel(d,'Cost');
  title(d,'Fitting error rises; the penalty falls');
  legend(d,{'Fitting error','alpha |T|','Total','Current tree','Minimum'},'Location','northoutside','Orientation','horizontal','FontSize',9);
  set(status,'String',sprintf('T_%d: |T| = %d. Fitting error %.3f + regularizer %.3f = total %.3f.   alpha = %.3f',stage-1,sizes(stage),risks(stage),alpha*sizes(stage),total(stage),alpha));
  if stage<numel(path)
   increase=risks(stage+1)-risks(stage); reduction=sizes(stage)-sizes(stage+1);
   set(notes,'String',{sprintf('Next collapse reduces |T| by %d: fitting error +%.3f; penalty -%.3f; total change %+.3f.',reduction,increase,alpha*reduction,increase-alpha*reduction), ...
    sprintf('Preview highlights the branch and its regions. Error increase per removed leaf: %.3f. Region labels are renumbered after merging.',increase/reduction)});
  else
   set(notes,'String',{'One leaf remains: all observations share one fitted mean C_1.', ...
    'Change alpha and click Best for alpha. This selects a subtree of the initial tree for the penalty you chose.'});
  end
  set(prev,'Enable',onoff(stage>1)); set(peek,'Enable',onoff(stage<numel(path))); set(next,'Enable',onoff(stage<numel(path)));
  set(alphaEdit,'String',sprintf('%.4g',alpha));
  set(slider,'Max',max(alpha,1.12*thresholds(end)),'Value',alpha); drawnow;
 end
end
function [X,y,T]=example()
% Fixed five-leaf topology: x1<s1; left x2<s2; right x1<s3;
% far-right x2<s4. Responses have deliberately different region means.
s=[.42 .55 .73 .45]; bounds=[0 s(1) 0 s(2);0 s(1) s(2) 1;s(1) s(3) 0 1;s(3) 1 0 s(4);s(3) 1 s(4) 1];
counts=[10 14 12 8 16]; means=[.2 .35 1.2 2.2 2.35]; X=[]; y=[];
for k=1:5
 n=counts(k); u=mod((1:n)*sqrt(2),1); v=mod((1:n)*sqrt(3),1); z=.12*sin((1:n)*1.7); z=z-mean(z);
 B=bounds(k,:);
 X=[X [B(1)+(B(2)-B(1))*(.08+.84*u);B(3)+(B(4)-B(3))*(.08+.84*v)]]; %#ok<AGROW>
 y=[y means(k)+z]; %#ok<AGROW>
end
T=newNode(1:numel(y),[0 1 0 1],0,y); T=split(T,1,1,s(1),1,X,y);
T=split(T,2,2,s(2),2,X,y); T=split(T,3,1,s(3),3,X,y); T=split(T,7,2,s(4),4,X,y);
end
function n=newNode(idx,B,depth,y)
v=y(idx); C=mean(v);
n=struct('idx',idx,'B',B,'depth',depth,'C',C,'SSE',sum((v-C).^2),'left',0,'right',0,'j',0,'s',NaN,'splitNumber',0);
end
function T=split(T,id,j,s,number,X,y)
n=T(id); left=X(j,n.idx)<s; B1=n.B; B2=n.B; B1(2*j)=s; B2(2*j-1)=s;
l=numel(T)+1; r=l+1; T(l)=newNode(n.idx(left),B1,n.depth+1,y); T(r)=newNode(n.idx(~left),B2,n.depth+1,y);
T(id).left=l; T(id).right=r; T(id).j=j; T(id).s=s; T(id).splitNumber=number;
end
function ids=leaves(T,id)
if nargin<2, id=1; end
if T(id).left==0, ids=id; else, ids=[leaves(T,T(id).left) leaves(T,T(id).right)]; end
end
function ids=active(T,id)
if nargin<2, id=1; end
ids=id; if T(id).left>0, ids=[ids active(T,T(id).left) active(T,T(id).right)]; end
end
function value=risk(T), ids=leaves(T); value=sum([T(ids).SSE]); end
function [path,removed,thresholds]=prune(T)
path={T}; removed={}; thresholds=0;
while T(1).left>0
 ids=active(T); branches=[]; gains=[];
 for id=ids
  if T(id).left==0, continue; end
  ll=leaves(T,id); branches(end+1)=id; gains(end+1)=(T(id).SSE-sum([T(ll).SSE]))/(numel(ll)-1); %#ok<AGROW>
 end
 g=min(gains); candidates=branches(gains<=g+1e-10*max(1,abs(g))); collapsed=[];
 for id=candidates
  if ~ismember(id,active(T)), continue; end
  T(id).left=0; T(id).right=0; collapsed(end+1)=id; %#ok<AGROW>
 end
 removed{end+1}=collapsed; path{end+1}=T; thresholds(end+1)=g; %#ok<AGROW>
end
end
function color=shade(C)
map=parula(128); color=map(max(1,min(128,1+round(127*C/2.6))),:);
end
function drawRegions(ax,T,X,y,marked,purple)
ll=leaves(T);
for k=1:numel(ll)
 n=T(ll(k)); B=n.B;
 patch(ax,B([1 2 2 1]),B([3 3 4 4]),shade(n.C),'FaceAlpha',.25,'EdgeColor',[.3 .3 .3]);
 text(ax,B(1)+.015,B(4)-.04,sprintf('R_%d',k),'FontSize',11);
end
scatter(ax,X(1,:),X(2,:),20,y,'filled'); colormap(ax,parula(128)); caxis(ax,[0 2.6]);
for id=marked(:).'
 if id==0, continue; end
 B=T(id).B; rectangle(ax,'Position',[B(1) B(3) B(2)-B(1) B(4)-B(3)],'EdgeColor',purple,'LineWidth',3);
end
xlabel(ax,'x_1'); ylabel(ax,'x_2'); axis(ax,[0 1 0 1]); axis(ax,'square');
cb=colorbar(ax,'southoutside'); cb.Label.String='Response y / prediction C_m'; cb.FontSize=9;
end
function drawSurface(ax,T,marked,purple)
for id=leaves(T)
 n=T(id); B=n.B; edge=[.3 .3 .3]; width=1;
 for root=marked(:).', if root>0 && ismember(id,leaves(T,root)), edge=purple; width=3; end; end
 patch(ax,B([1 2 2 1]),B([3 3 4 4]),n.C*ones(1,4),shade(n.C),'FaceAlpha',.85,'EdgeColor',edge,'LineWidth',width);
end
xlabel(ax,'x_1'); ylabel(ax,'x_2'); zlabel(ax,'C_m'); axis(ax,[0 1 0 1 0 2.6]); view(ax,[-35 28]);
end
function drawTree(ax,T,marked,purple)
ids=active(T); ll=leaves(T); x=zeros(1,numel(T));
for k=1:numel(ll), x(ll(k))=(k-.5)/numel(ll); end
for id=fliplr(ids), if T(id).left>0, x(id)=(x(T(id).left)+x(T(id).right))/2; end; end
highlight=[];
for root=marked(:).', if root>0, highlight=[highlight active(T,root)]; end; end %#ok<AGROW>
for id=ids
 n=T(id); yy=1-n.depth*.26;
 for child=[n.left n.right]
  if child==0, continue; end
  color=[.5 .5 .5]; width=1; if ismember(id,highlight), color=purple; width=2; end
  plot(ax,[x(id) x(child)],[yy 1-T(child).depth*.26],'Color',color,'LineWidth',width);
 end
end
for id=ids
 n=T(id); edge=[.45 .45 .45]; if ismember(id,highlight), edge=purple; end
 if n.left>0, str=sprintf('x_%d < s_%d\ns_%d = %.2f',n.j,n.splitNumber,n.splitNumber,n.s); color=[.96 .96 .96];
 else, str=sprintf('R_%d\nC=%.2f',find(ll==id),n.C); color=.75+.25*shade(n.C); end
 text(ax,x(id),1-n.depth*.26,str,'HorizontalAlignment','center','FontSize',10,'BackgroundColor',color,'EdgeColor',edge,'Margin',3);
end
axis(ax,[-.10 1.1 -.05 1.15]); axis(ax,'off');
text(ax,.5,-.02,'Left: yes     Right: no','FontSize',10,'HorizontalAlignment','center');
end
function s=onoff(v), if v, s='on'; else, s='off'; end; end
function pairs=allCosts(T,id)
if nargin<2, id=1; end
pairs=[T(id).SSE 1]; if T(id).left==0, return; end
L=allCosts(T,T(id).left); R=allCosts(T,T(id).right);
for i=1:size(L,1), for j=1:size(R,1), pairs(end+1,:)=L(i,:)+R(j,:); end; end %#ok<AGROW>
end
function result=selftest()
[~,y,T]=example(); [P,removed,g]=prune(T); costs=cellfun(@risk,P); sizes=cellfun(@(t)numel(leaves(t)),P);
assert(sizes(1)==5 && sizes(end)==1 && all(diff(sizes)<0));
assert(all(diff(costs)>=-1e-10) && all(diff(g)>=-1e-10));
for k=1:numel(P)
 t=P{k}; ll=leaves(t); combined=[];
 for id=ll
  n=t(id); combined=[combined n.idx]; %#ok<AGROW>
  assert(abs(n.C-mean(y(n.idx)))<1e-12);
  assert(abs(n.SSE-sum((y(n.idx)-n.C).^2))<1e-12);
 end
 assert(isequal(sort(combined),1:numel(y)));
 if k<numel(P)
  assert(~isempty(removed{k}));
  assert(abs((costs(k+1)-costs(k))/(sizes(k)-sizes(k+1))-g(k+1))<1e-9);
 end
end
pairs=allCosts(T);
for alpha=unique([0 g linspace(0,1.2*g(end),200)])
 assert(abs(min(costs+alpha*sizes)-min(pairs(:,1)+alpha*pairs(:,2)))<1e-8);
end
assert(abs(P{end}(1).C-mean(y))<1e-12);
result=struct('passed',true,'leafCounts',sizes,'criticalAlpha',g,'allSubtreesChecked',size(pairs,1)); disp(result);
end
