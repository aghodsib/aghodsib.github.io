function out=random_forest_demo(mode)
% RANDOM_FOREST_DEMO  Samples, random feature choices, and majority votes.
% Run random_forest_demo. Base MATLAB R2016b+, no extra toolboxes.
% Run random_forest_demo('selftest') for numerical checks.
% X is d-by-n: rows are features, columns are observations.
% Start with Identical trees; add trees. Then select Bagging or Random forest.
% Click either decision plot to select a query. Inspect tree/node menus to
% see which original columns reached a node and which feature rows were
% eligible there. Bootstrap samples retain repeated columns.
% Trees use Gini splits, max depth 4 and minimum child size 2 (counting
% bootstrap multiplicity). Random forests consider one of two features
% afresh at EACH node. A zero-gain split is allowed if children are valid.
% Leaf ties and ensemble vote ties choose class 0. Votes are hard classes,
% not calibrated probabilities. This small dataset illustrates mechanisms;
% it does not claim random forests always improve accuracy.
if nargin<1, mode='gui'; end
if strcmpi(mode,'selftest'), out=checks(); return; end
if ~strcmpi(mode,'gui'), error('Use gui or selftest.'); end
r=RandStream('mt19937ar','Seed',841);
X=.02+.96*rand(r,2,48); y=double(X(2,:)>.48+.22*sin(2*pi*X(1,:)));
flip=rand(r,1,48)<.08; y(flip)=1-y(flip);
kind=1; count=1; current=1; nodeId=1; query=[.5;.5]; trees={}; gridVotes=[];
[g1,g2]=meshgrid(linspace(0,1,90)); G=[g1(:).';g2(:).'];
blue=[.12 .40 .78]; orange=[.90 .38 .10]; bg=[.97 .98 1];
f=figure('Name','Random forests: different trees, one vote','NumberTitle','off', ...
 'Units','normalized','Position',[.025 .05 .95 .86],'Color',bg); out=f;
label([.04 .94 .92 .04],'Random forests: different trees, one vote',21);
info=label([.04 .875 .92 .055],'',12);
a=axes('Parent',f,'Position',[.06 .57 .42 .21]);
b=axes('Parent',f,'Position',[.57 .55 .37 .28]);
c=axes('Parent',f,'Position',[.06 .20 .39 .28]);
d=axes('Parent',f,'Position',[.57 .20 .37 .28]);
method=uicontrol(f,'Style','popupmenu','Units','normalized','Position',[.04 .065 .27 .045], ...
 'String',{'1. Identical trees','2. Bagging: sample columns','3. Random forest: columns + rows'},'FontSize',11,'Callback',@changeMethod);
add=uicontrol(f,'Style','pushbutton','Units','normalized','Position',[.33 .065 .14 .045], ...
 'String','Add one tree','FontSize',11,'Callback',@addTree);
uicontrol(f,'Style','pushbutton','Units','normalized','Position',[.49 .065 .13 .045], ...
 'String','Reset to 1 tree','FontSize',10,'Callback',@reset);
treeMenu=uicontrol(f,'Style','popupmenu','Units','normalized','Position',[.65 .065 .13 .045],'String',{'Tree 1'},'Callback',@pickTree);
nodeMenu=uicontrol(f,'Style','popupmenu','Units','normalized','Position',[.80 .065 .16 .045],'String',{'Root'},'Callback',@pickNode);
footer=label([.04 .005 .93 .045],'',10);
rebuild(); render();
 function h=label(pos,str,sz)
  h=uicontrol(f,'Style','text','Units','normalized','Position',pos,'String',str,'FontSize',sz,'BackgroundColor',bg,'HorizontalAlignment','left');
 end
 function changeMethod(src,~), kind=get(src,'Value'); current=count; nodeId=1; rebuild(); render(); end
 function addTree(~,~)
  if count>=25, return; end
  count=count+1; current=count; nodeId=1; append(); render();
 end
 function reset(~,~), count=1; current=1; nodeId=1; rebuild(); render(); end
 function pickTree(src,~), current=get(src,'Value'); nodeId=1; render(); end
 function pickNode(src,~), nodeId=get(src,'Value'); render(); end
 function click(src,~)
  p=get(src,'CurrentPoint'); query=max(0,min(1,p(1,1:2).')); render();
 end
 function rebuild()
  trees={}; gridVotes=[];
  for k=1:count, append(); end
 end
 function append()
  k=numel(trees)+1; stream=RandStream('mt19937ar','Seed',1000+k);
  if kind==1, ids=1:size(X,2); else, ids=randi(stream,size(X,2),1,size(X,2)); end
  m=2; if kind==3, m=1; end
  trees{k}=trainTree(X,y,ids,m,stream);
  gridVotes(k,:)=classify(trees{k},G);
 end
 function render()
  for ax=[a b c d], cla(ax,'reset'); hold(ax,'on'); set(ax,'FontSize',10,'Box','on'); end
  T=trees{current}; nodeId=min(nodeId,numel(T)); n=T(nodeId);
  names=arrayfun(@(k)sprintf('Tree %d',k),1:count,'UniformOutput',false);
  set(treeMenu,'String',names,'Value',current);
  names=arrayfun(@(k)sprintf('Node %d (depth %d)',k,T(k).depth),1:numel(T),'UniformOutput',false);
  set(nodeMenu,'String',names,'Value',nodeId); set(add,'Enable',enabled(count<25));
  % The matrix is in ORIGINAL column order. Multiplicities underneath
  % explicitly show which columns were duplicated or excluded.
  ids=n.ids; copies=accumarray(ids(:),1,[size(X,2) 1]).';
  im=imagesc(a,X); set(im,'AlphaData',.15+.85*repmat(copies>0,2,1));
  colormap(a,gray(128)); caxis(a,[0 1]);
  for row=n.candidates
   rectangle(a,'Position',[.5 row-.48 size(X,2) .96],'EdgeColor',[.10 .65 .30],'LineWidth',2);
  end
  for i=1:size(X,2)
   text(a,i,2.9,num2str(copies(i)),'HorizontalAlignment','center','FontSize',8,'Color',[.1 .1 .1]);
  end
  set(a,'YDir','reverse','YTick',[1 2 2.9],'YTickLabel',{'x_1','x_2','copies'},'XTick',[1 8 16 24 32 40 48]);
  axis(a,[.5 size(X,2)+.5 .4 3.25]); xlabel(a,'Original column index i (one observation per column)');
  title(a,sprintf('X: 2 feature rows, 48 observation columns | tree %d, node %d',current,nodeId),'FontSize',11);
  plotMap(b,gridVotes(current,:)); title(b,sprintf('Tree %d: its own decision regions',current));
  plotMap(c,double(mean(gridVotes,1)>.5)); title(c,sprintf('Majority vote of %d trees',count));
  % Ring the observations used by this tree. Larger rings mean more copies.
  ct=accumarray(T(1).ids(:),1,[size(X,2) 1]); used=ct>0;
  scatter(b,X(1,used),X(2,used),30+20*ct(used),'k','LineWidth',.7,'HitTest','off');
  votes=cellfun(@(t)classify(t,query),trees); p=sum(votes); prediction=double(p>count/2);
  for k=1:count
   color=blue; if votes(k)==1, color=orange; end
   patch(d,[k-.4 k+.4 k+.4 k-.4],[.58 .58 .94 .94],color,'EdgeColor','w');
   text(d,k,.76,num2str(votes(k)),'Color','w','HorizontalAlignment','center','FontWeight','bold');
  end
  xlim(d,[.4 max(5,count)+.6]); ylim(d,[0 1.1]); set(d,'YTick',[],'XTick',1:count); xlabel(d,'Tree number');
  title(d,sprintf('Votes for query (%.2f, %.2f)',query(1),query(2)));
  text(d,.03,.32,sprintf('Class 0: %d votes     Class 1: %d votes',count-p,p),'Units','normalized','FontSize',12);
  text(d,.03,.12,sprintf('Forest prediction: class %d',prediction),'Units','normalized','FontSize',14,'FontWeight','bold');
  rootCopies=accumarray(T(1).ids(:),1,[size(X,2) 1]);
  if n.left>0, split=sprintf('Node %d chooses x_%d < %.2f.',nodeId,n.j,n.s);
  else, split=sprintf('Node %d is a leaf: predicts class %d.',nodeId,n.value); end
  set(info,'String',{sprintf('Tree %d uses %d distinct columns from %d draws. %s',current,sum(rootCopies>0),numel(T(1).ids),split), ...
   'Matrix: faded columns are absent here; copies counts repetitions; green outlines mark candidate feature rows at this node.'});
  explanations={ ...
   'Identical data + identical split choices give identical trees. Adding copies leaves the decision boundary unchanged.', ...
   'Bagging draws 48 columns WITH replacement for each tree. Both feature rows are considered at every split.', ...
   'Random forest: bootstrap columns for each tree; choose ONE feature row afresh at EACH split. Inspect different nodes.'};
  set(footer,'String',{explanations{kind},'Click either decision plot to choose a query. Blue = class 0; orange = class 1. Vote ties choose class 0.'});
  drawnow;
 end
 function plotMap(ax,pred)
  image(ax,[0 1],[0 1],reshape(pred,size(g1)),'CDataMapping','scaled');
  colormap(ax,[.86 .92 1;1 .90 .82]); caxis(ax,[0 1]); set(ax,'YDir','normal');
  scatter(ax,X(1,y==0),X(2,y==0),22,blue,'filled');
  scatter(ax,X(1,y==1),X(2,y==1),22,orange,'filled');
  plot(ax,query(1),query(2),'kp','MarkerSize',14,'MarkerFaceColor',[1 1 .2]);
  xlabel(ax,'x_1'); ylabel(ax,'x_2'); axis(ax,[0 1 0 1]);
  set(get(ax,'Children'),'HitTest','off'); set(ax,'ButtonDownFcn',@(src,event)click(src,event));
 end
end
function T=trainTree(X,y,ids,m,stream)
T=struct('ids',{},'depth',{},'candidates',{},'j',{},'s',{},'left',{},'right',{},'value',{});
visit(ids,0);
 function id=visit(indices,depth)
  id=numel(T)+1; v=y(indices);
  n=struct('ids',indices,'depth',depth,'candidates',[],'j',0,'s',NaN,'left',0,'right',0,'value',double(mean(v)>.5));
  T(id)=n;
  if depth>=4 || numel(v)<4 || all(v==v(1)), return; end
  if m==size(X,1), rows=1:size(X,1); else, rows=randperm(stream,size(X,1),m); end
  T(id).candidates=rows; best=Inf; chosen=0; threshold=NaN;
  for j=rows
   values=unique(X(j,indices)); thresholds=(values(1:end-1)+values(2:end))/2;
   for s=thresholds(:).'
    left=X(j,indices)<s; a=v(left); b=v(~left);
    if numel(a)<2 || numel(b)<2, continue; end
    cost=numel(a)*2*mean(a)*(1-mean(a))+numel(b)*2*mean(b)*(1-mean(b));
    if cost<best-1e-12, best=cost; chosen=j; threshold=s; end
   end
  end
  if chosen==0, return; end
  left=X(chosen,indices)<threshold;
  T(id).j=chosen; T(id).s=threshold;
  l=visit(indices(left),depth+1); rr=visit(indices(~left),depth+1);
  T(id).left=l; T(id).right=rr;
 end
end
function y=classify(T,X)
y=zeros(1,size(X,2));
for k=1:size(X,2)
 id=1;
 while T(id).left>0
  if X(T(id).j,k)<T(id).s, id=T(id).left; else, id=T(id).right; end
 end
 y(k)=T(id).value;
end
end
function s=enabled(v)
if v, s='on'; else, s='off'; end
end
function result=checks()
X=[0 .1 .2 .3 .7 .8 .9 1; .2 .8 .3 .7 .1 .9 .4 .6]; y=[0 0 0 0 1 1 1 1];
r=RandStream('mt19937ar','Seed',10); a=trainTree(X,y,1:8,2,r);
b=trainTree(X,y,1:8,2,RandStream('mt19937ar','Seed',99));
assert(isequaln(a,b)); assert(isequal(classify(a,X),y));
assert(classify(a,[a(1).s;.5])==1);
ids=[1 1 1 2 5 6 6 8]; t=trainTree(X,y,ids,1,r);
assert(isequal(t(1).ids,ids));
for i=1:numel(t)
 n=t(i); assert(numel(n.candidates)<=1);
 if n.left>0
  assert(ismember(n.j,n.candidates));
  assert(all(X(n.j,t(n.left).ids)<n.s));
  assert(all(X(n.j,t(n.right).ids)>=n.s));
  assert(isequal(sort([t(n.left).ids t(n.right).ids]),sort(n.ids)));
 end
end
votes=[0 1 1;1 1 0;1 0 0]; assert(isequal(double(mean(votes,1)>.5),[1 1 0]));
assert(double(mean([0 1])>.5)==0);
result=struct('passed',true); disp('Passed: identical trees, split routing, bootstrap multiplicity, candidate features, and voting.');
end
