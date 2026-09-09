
        uint lane=thread_position_in_grid.x,row0=thread_position_in_grid.y*ROWS,batch=thread_position_in_grid.z;
        float totals[ROWS]={0};
        const device ushort* packed=reinterpret_cast<const device ushort*>(q);
        for(uint k=lane*VALUES;k<K;k+=32*VALUES){
            float scaled[VALUES],sums[5]={0,0,0,0,0};
            for(uint j=0;j<VALUES;j+=4){
                float a=float(x[batch*K+k+j]),b=float(x[batch*K+k+j+1]);
                float c=float(x[batch*K+k+j+2]),d=float(x[batch*K+k+j+3]);
                scaled[j]=a;scaled[j+1]=b/16;scaled[j+2]=c/256;scaled[j+3]=d/4096;
                if(NATIVE_SUM)sums[0]+=float(x[batch*K+k+j]+x[batch*K+k+j+1]+x[batch*K+k+j+2]+x[batch*K+k+j+3]);
                else sums[0]+=a+b+c+d;
                if(WALSH){
                    sums[1]+=a-b+c-d;sums[2]+=a+b-c-d;sums[3]+=a-b-c+d;
                    sums[4]+=(j&4)?-(a-b-c+d):(a-b-c+d);
                }
            }
            for(uint r=0;r<ROWS;r++){
                uint row=row0+r;if(row>=N)continue;
                uint group=row*(K/G)+k/G;float dot=0;
                for(uint j=0;j<VALUES;j+=4){
                    ushort word=packed[row*(K/4)+(k+j)/4];
                    if(ROW_GAIN)word^=0x8888u;
                    dot+=scaled[j]*float(word&15)+scaled[j+1]*float(word&240)+scaled[j+2]*float(word&3840)+scaled[j+3]*float(word&61440);
                }
                
            uint bit=group*(B+1),idx=bit/32,shift=bit%32;
            ulong pair=(ulong(fields[idx+1])<<32)|ulong(fields[idx]);
            uint field=uint(pair>>shift),mag=(field&((1u<<B)-1))+bases[0],sign=(field>>B)&1;
            uint chunk=group/32,l=group%32,mask=index[chunk*2];int delta=0;
            if((mask>>l)&1){uint rank=popcount(mask&((1u<<l)-1));delta=int(residual[index[chunk*2+1]+rank])+as_type<int>(bases[1]);}
            float scale=as_type<float>((mag|(sign<<15))<<16);
            float bias=as_type<float>((uint(int(mag)+384+delta)|((sign^1)<<15))<<16);
            totals[r]+=scale*dot+bias*sums[0];
        
            }
        }
        for(uint r=0;r<ROWS;r++){
            totals[r]=simd_sum(totals[r]);
            
            if(lane==0&&row0+r<N)y[batch*N+row0+r]=T(totals[r]);
        }
    