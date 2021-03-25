clear;clc;close all
h = figure();
set(gcf,'name','Gravity Simulation','NumberTitle','off','color',[0.1 0.1 0.1],'units','normalized','position',[0.2 0.15 0.4 0.6],'menubar','none','toolbar','none');
number_of_points = 5000;
dimension = 2;
gamma = 0.005;
write_video = 0;
video_name = 'simulation.avi';

% theta = linspace(0,2*pi,number_of_points+1);
% theta(end) = [];
% for i = 1:number_of_points
%     r(i,:) = 5*[cos(theta(i)) sin(theta(i))]+rand(1,2);
% end

%rng('default');
r = rand(number_of_points,dimension);
v = zeros(size(r,1),dimension);
m = ones(size(r,1),1);

epsilon = 2*gamma;

x_lim = [min(r(:,1)) max(r(:,1))];
y_lim = [min(r(:,2)) max(r(:,2))];
counter = 0;
if write_video == 1
    writerObj = VideoWriter(video_name);
    writerObj.FrameRate = 10;
    open(writerObj);
end
subplot = @(m,n,p) subtightplot (m, n, p, [0 0], [0 0], [0 0]); 
subplot(1,1,1)
while ishandle(h)
    counter = counter+1;
    
    index = find_nearest_points(r,epsilon);
    if isempty(index)~=1
        [r,v,m] = replace_nearest_points(r,v,m,index);
    end
    
    if size(r,1)>1
        f = gravity(r,m,number_of_points);
        a = f./m;        
        a_norm = calculate_norm_vector(a);        
        [~,I] = maxk(a_norm,10);
        
        for i = 1:length(I)
            dt(i) = find_dt(a(I(i),:),v(I(i),:),gamma);
        end        
        dt = min(dt);
                
        r = 0.5*a.*dt^2+v*dt+r;  
        v = a.*dt+v;
        
    end
    
    clustering_time(counter) = dt; 
    number_of_frames(counter) = counter;
    number_of_clusters(counter) = size(r,1);
      
    ax = gca;cla(ax);    
    scatter(r(:,1),r(:,2),m,'m','filled','MarkerFaceAlpha',0.6)                 
    pbaspect([1 1 1])
    xlim(x_lim)
    ylim(y_lim) 
    %title({'',['Number of Particles = ',num2str(size(r,1))],''},'interpreter','latex','color','w')
    set(gca,'color',[0.1 0.1 0.1]);
    axis equal    
    axis off
    drawnow 
    if write_video==1
        video_frame = getframe(gcf);
        writeVideo(writerObj, video_frame);
    end
end
if write_video ==1
    close(writerObj);
end
figure();
set(gcf,'name','Gravity Simulation','NumberTitle','off','color','w','units','normalized','position',[0.5 0.42 0.3 0.4]);
plot(number_of_frames,clustering_time,'b')
xlabel('Frame Number','interpreter','latex','fontsize',14)
ylabel('dt','interpreter','latex','fontsize',14)
xlim([1 max(number_of_frames)])
pbaspect([1 1 1])
set(gca,'color','w','TickDir','out','box','on','BoxStyle','full','XColor','k','YColor','k','TickLabelInterpreter','latex','fontsize',14);

figure();
set(gcf,'name','Gravity Simulation','NumberTitle','off','color','w','units','normalized','position',[0.5 0.42 0.3 0.4]);
plot(number_of_frames,number_of_clusters,'b')
pbaspect([1 1 1])
xlabel('Frame Number','interpreter','latex','fontsize',14)
ylabel('Number of Clusters','interpreter','latex','fontsize',14)
xlim([1 max(number_of_frames)])
set(gca,'color','w','TickDir','out','box','on','BoxStyle','full','XColor','k','YColor','k','TickLabelInterpreter','latex','fontsize',14);

function force = gravity(r,m,N)
idx = knnsearch(r,r,'k',N);
for i = 1:size(r,1)    
    force(i,:) = calculate_gravitational_force(r(idx(i,:),:),m(idx(i,:)));
end
    function sum_force = calculate_gravitational_force(r,m)
        r = r-r(1,:);
        r(1,:) = [];
        M = m(1);
        m(1) = [];
        distance = vecnorm(r')';
        f = (6.6743e-5*M.*m)./(distance.^2);
        r_norm = r./distance;
        force_vector = r_norm.*f;
        sum_force = sum(force_vector,1);
    end
end

function t = find_dt(a,v,gamma)
opts = optimset('Display','off');
fgfit=@(t) calculate_norm_g(t,a,v)-gamma;
lb=0;
ub=10;
t0 = 1;
t = lsqnonlin(fgfit,t0,lb,ub,opts);
end

function g_norm = calculate_norm_g(t,a,v)
g = 0.5*a.*(t.^2)+v.*t;
g_norm = calculate_norm_vector(g);
end

function vec_norm = calculate_norm_vector(vec)
vec_norm = vecnorm(vec')';
end

function index = find_nearest_points(r,epsilon)
neighbors = rangesearch(r,r,epsilon);
for i = 1:length(neighbors)
    if length(neighbors{i})==1
        neighbors{i}(1) = [];
    end
end

index =[];
counter = 0;
used_points = zeros(length(neighbors),1);
if isempty(neighbors)~=1    
    for i =1:length(neighbors)
        if ~used_points(i)
            seed = neighbors{i};
            if ~isempty(seed)
                size_one = 0;
                size_two = length(seed);
                while size_two~=size_one
                    size_one = length(seed);
                    idx = neighbors(seed);
                    idx = horzcat(idx{:});
                    idx = unique(idx);
                    if ~any(intersect(idx,seed))
                        seed = sort([idx;seed]);
                    else
                        seed = idx;
                    end
                    size_two = length(seed);
                end
                used_points(seed) = 1;
                counter = counter+1;
                index{counter,1} = seed;
            end
        end
    end
end
end

function [r,v,m] = replace_nearest_points(r,v,m,index)
all_idx = horzcat(index{:});
not_in_bundle = setdiff(1:size(r,1),all_idx);
for i = 1:length(index)
    replaced_r(i,:) = sum(m(index{i}).*r(index{i},:))./sum(m(index{i}));
    replaced_v(i,:) = sum(m(index{i}).*v(index{i},:))./sum(m(index{i}));
    replaced_m(i,1) = sum(m(index{i}));    
end
r = [r(not_in_bundle,:);replaced_r];
v = [v(not_in_bundle,:);replaced_v];
m = [m(not_in_bundle);replaced_m];
end

function h=subtightplot(m,n,p,gap,marg_h,marg_w,varargin)
if (nargin<4) || isempty(gap),    gap=0.01;  end
if (nargin<5) || isempty(marg_h),  marg_h=0.05;  end
if (nargin<5) || isempty(marg_w),  marg_w=marg_h;  end
if isscalar(gap),   gap(2)=gap;  end
if isscalar(marg_h),  marg_h(2)=marg_h;  end
if isscalar(marg_w),  marg_w(2)=marg_w;  end
gap_vert   = gap(1);
gap_horz   = gap(2);
marg_lower = marg_h(1);
marg_upper = marg_h(2);
marg_left  = marg_w(1);
marg_right = marg_w(2);
%note n and m are switched as Matlab indexing is column-wise, while subplot indexing is row-wise :(
[subplot_col,subplot_row]=ind2sub([n,m],p);  
% note subplot suppors vector p inputs- so a merged subplot of higher dimentions will be created
subplot_cols=1+max(subplot_col)-min(subplot_col); % number of column elements in merged subplot 
subplot_rows=1+max(subplot_row)-min(subplot_row); % number of row elements in merged subplot   
% single subplot dimensions:
%height=(1-(m+1)*gap_vert)/m;
%axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh; 
height=(1-(marg_lower+marg_upper)-(m-1)*gap_vert)/m;
%width =(1-(n+1)*gap_horz)/n;
%axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;
width =(1-(marg_left+marg_right)-(n-1)*gap_horz)/n;
% merged subplot dimensions:
merged_height=subplot_rows*( height+gap_vert )- gap_vert;
merged_width= subplot_cols*( width +gap_horz )- gap_horz;
% merged subplot position:
merged_bottom=(m-max(subplot_row))*(height+gap_vert) +marg_lower;
merged_left=(min(subplot_col)-1)*(width+gap_horz) +marg_left;
pos_vec=[merged_left merged_bottom merged_width merged_height];
% h_subplot=subplot(m,n,p,varargin{:},'Position',pos_vec);
% Above line doesn't work as subplot tends to ignore 'position' when same mnp is utilized
h=subplot('Position',pos_vec,varargin{:});
if (nargout < 1),  clear h;  end
end