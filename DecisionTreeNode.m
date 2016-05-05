classdef DecisionTreeNode < handle
    %DecisionTreeNode class: Nodes of entropy-based binary decision tree.
    %   For 24-787 HW1, you need to complete the following functions in
    %   this file: find_decision_attrib, train, classify, entropy, and
    %   entropy_of_class.
    
    properties
        decision_attrib;        % the index of the attrib where split occurs
        available_attribs;      % the indices of available attributes; 1xn vector
        decision;               % the class for all input data, if no split occurs
        left_node;              % handle to the left leaf
        right_node;             % handle to the right leaf
        parent_node;            % handle to the parent leaf
    end
    
    methods
        function this = DecisionTreeNode()
            this.decision_attrib = -1;
            this.decision = -1;
            this.parent_node = [];
            this.available_attribs = [];
        end
        
        
        function find_decision_attrib(this,attrib,class)
            
           
            info_gain = zeros(size(this.available_attribs));
            index=1;
            
       
            for i=this.available_attribs
                
                %l is the total number of test cases
                l=length(attrib(:,i));
                countleft=0;
                countright=0;
                classleft=[];
                classright=[];
                for j=1:l
                    
                    if attrib(j,i)==1
                        %disp('Here');
                        countleft=countleft+1;
                        classleft(countleft)=class(j,1);
                    else
                        %disp('Here2');
                        countright=countright+1;
                        classright(countright)=class(j,1);
                    end
                end
                
               
                l=countleft+countright;
                pl=countleft/l;
                pr=countright/l;
                
                
                H= pl*DecisionTreeNode.entropy_of_class(classleft) + pr*DecisionTreeNode.entropy_of_class(classright);
                info_gain(index)=entropy([pl pr]) - H;
            
                index=index+1;
                
            end
            
            
            [~,I]=max(info_gain);
            this.decision_attrib=this.available_attribs(I);
            
            
        end


        %'this' is the current node. Attrib is an array containing attributes
        %and their values. This is same as the structure of train_attrib.
        %It is a matrix of dimension 1000x15 where 15 is the number of
        %attributes and 100 is the training samples. Class is the final
        %value.
        
        function train(this,attrib,class)
            
            %attribsize=size(attrib)
            
            find_decision_attrib(this,attrib,class);
            j=this.decision_attrib;
            
            %possibility=this.available_attribs    
            countleft=0;
            countright=0;
            
            classleft=[];
            classright=[];
            attribleft=[];
            attribright=[];
            
            l=length(attrib(:,j));

            %MAKING SOME GENERAL CALCULATIONS OF THE LEFT AND RIGHT DIVISIONS 

            for i=1:l
                if attrib(i,j)==1
                    countleft=countleft+1;
                    attribleft(countleft,:)=attrib(i,:);
                    classleft(countleft,1)=class(i,1);
                else
                    countright=countright+1;
                    attribright(countright,:)=attrib(i,:);
                    classright(countright,1)=class(i,1);
                end
            end

            %CHECKING FOR PERFECT SPLIT
            if length(unique(class))==1
                %disp('Perfect Split');
                this.decision=unique(class);
                this.decision_attrib=-1;
                return;
            end
            
            %CHECKING FOR OVERFLOW OF ATTRIBUTES

            if isempty(this.available_attribs)
                %disp('No more available attributes. Making a decision...\n');
                joycount=sum(classleft==1)+sum(classright==1);
                sadcount=sum(classleft==0)+sum(classright==0);

                if joycount>=sadcount
                    this.decision=1;
                else
                    this.decision=0;
                end
                return;
            end
                
            %IF IT CAN BE SPLIT FURTHER
            
            %Creating a left and right node
            this.left_node=DecisionTreeNode();
            this.right_node=DecisionTreeNode();
            this.left_node.parent_node=this;
            this.right_node.parent_node=this;
            
            this.left_node.available_attribs=this.available_attribs(this.available_attribs~=this.decision_attrib);
            this.right_node.available_attribs=this.available_attribs(this.available_attribs~=this.decision_attrib);
%             train(this.left_node,attribleft,classleft);
%              train(this.right_node,attribright,classright);
            if ~isempty(classleft)
                
                %this.left_node.available_attribs
                %disp('going left')
                train(this.left_node,attribleft,classleft);
                %disp('Done left');
            end
            
            if ~isempty(classright)
                
                %this.right_node.available_attribs
                %disp('Going right');
                train(this.right_node,attribright,classright);
                %disp('Done right')
            end
            return;

        end  
        
        
        function class = classify(this,attrib)
            
            class = -ones(size(attrib,1),1); %initialize class labels to -1
            l=size(attrib,1);
            if (this.decision_attrib==-1)
                %disp('Reached a leaf... Returning');
                %this.decision
                class(:)=this.decision*ones(l,1);
                return;
            end
   
            
            j=this.decision_attrib;
            
            leftpartition=[];
            rightpartition=[];
            leftcount=0;
            rightcount=0;
            
            
            
            for i=1:l
                if attrib(i,j)==1
                    leftcount=leftcount+1;
                    leftpartition(leftcount,:)=attrib(i,:);
                    leftindices(leftcount)=i;
                else
                    rightcount=rightcount+1;
                    rightpartition(rightcount,:)=attrib(i,:);
                    rightindices(rightcount)=i;
                end
            end
            
            
            if (~isempty(rightpartition))
                %disp('Going 0:right');
                
                returnval=classify(this.right_node,rightpartition);
                %disp('Going 0:right..DONE');
            
            
                count=0;
                for i=rightindices
                    count=count+1;
                    class(i)=returnval(count);
                end
                
            end
                    
            if (~isempty(leftpartition))
                %disp('Going 1:left');
                
                returnval=classify(this.left_node,leftpartition);
                %disp('Going 1:left..DONE');
            
            
                count=0;
                for i=leftindices
                    count=count+1;
                    class(i)=returnval(count);
                end
            end
        % Label test data based on the given attributes
        % HINT: this will be a recursive function
        
            
            
            % Check to see if we are at a leaf node. If so, assign the correct label to the output.
            %\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            % INSERT CODE HERE
            %\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            
            
            % If we are not at leaf node, split the data and initiate recursion
            %\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            % INSERT CODE HERE
            %\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

            
        end
    end
    
    %% Static methods can be called without object instantiation
    methods (Static)
        function h = entropy(p)
            p=p/sum(p);
            h=0;
            for i=p
                if i==0
                    w=0;
                else w=log2(i);
                end
                    
                h=h-i*w;
            end
        end
        
        
        function h = entropy_of_class(class)
            % Compute the entropy of (h) given a binary vector of labels 
            % (class), in which Joy = 1 and Despair = 0
            L=length(class);
            count=0;
            for i=class
                if i==1
                    count=count+1;
                end
            end
            p1=count/L;
            p=[p1 1-p1];
            h=DecisionTreeNode.entropy(p);
            
        end
    end
end