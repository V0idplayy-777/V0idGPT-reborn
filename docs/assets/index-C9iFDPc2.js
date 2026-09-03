var e=Object.create,t=Object.defineProperty,n=Object.getOwnPropertyDescriptor,r=Object.getOwnPropertyNames,i=Object.getPrototypeOf,a=Object.prototype.hasOwnProperty,o=(e,t)=>()=>(t||(e((t={exports:{}}).exports,t),e=null),t.exports),s=(e,n)=>{let r={};for(var i in e)t(r,i,{get:e[i],enumerable:!0});return n||t(r,Symbol.toStringTag,{value:`Module`}),r},c=(e,i,o,s)=>{if(i&&typeof i==`object`||typeof i==`function`)for(var c=r(i),l=0,u=c.length,d;l<u;l++)d=c[l],!a.call(e,d)&&d!==o&&t(e,d,{get:(e=>i[e]).bind(null,d),enumerable:!(s=n(i,d))||s.enumerable});return e},l=(n,r,o)=>(o=n==null?{}:e(i(n)),c(r||!n||!n.__esModule||!a.call(n,`default`)?t(o,`default`,{value:n,enumerable:!0}):o,n));(function(){let e=document.createElement(`link`).relList;if(e&&e.supports&&e.supports(`modulepreload`))return;for(let e of document.querySelectorAll(`link[rel="modulepreload"]`))n(e);new MutationObserver(e=>{for(let t of e)if(t.type===`childList`)for(let e of t.addedNodes)e.tagName===`LINK`&&e.rel===`modulepreload`&&n(e)}).observe(document,{childList:!0,subtree:!0});function t(e){let t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),t.credentials=e.crossOrigin===`use-credentials`?`include`:e.crossOrigin===`anonymous`?`omit`:`same-origin`,t}function n(e){if(e.ep)return;e.ep=!0;let n=t(e);fetch(e.href,n)}})();var u=1e-7,d=1e-4,f=class{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}},p=class{refCount(e){return m(`refCount`)}incRef(e){return m(`incRef`)}timerAvailable(){return!0}time(e){return m(`time`)}read(e){return m(`read`)}readSync(e){return m(`readSync`)}readToGPU(e,t){return m(`readToGPU`)}numDataIds(){return m(`numDataIds`)}disposeData(e,t){return m(`disposeData`)}write(e,t,n){return m(`write`)}move(e,t,n,r,i){return m(`move`)}createTensorFromGPUData(e,t,n){return m(`createTensorFromGPUData`)}memory(){return m(`memory`)}floatPrecision(){return m(`floatPrecision`)}epsilon(){return this.floatPrecision()===32?u:d}dispose(){return m(`dispose`)}};function m(e){throw Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}function h(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,v(e,t,n)}function g(e,t,n){return Math.max(e,Math.min(t,n))}function _(e){return e%2==0?e:e+1}function v(e,t,n){let r=e[t];e[t]=e[n],e[n]=r}function y(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function b(e,t){if(!e)throw Error(typeof t==`string`?t:t())}function x(e,t,n=``){b(w(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function S(e){b(e!=null,()=>`The input to the tensor constructor must be a non-null value.`)}function C(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function w(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function T(e){return e%1==0}function E(e){let t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function D(e,t){return t<=e.length?e:e+` `.repeat(t-e.length)}function ee(e,t=e=>0,n,r){return new Promise((i,a)=>{let o=0,s=()=>{if(e()){i();return}o++;let c=t(o);if(n!=null&&o>=n){a();return}r==null?setTimeout(s,c):r(s,c)};s()})}function te(e,t){let n=1,r=-1;for(let t=0;t<e.length;++t)if(e[t]>=0)n*=e[t];else if(e[t]===-1){if(r!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${t}`);r=t}else if(e[t]<0)throw Error(`Shapes can not be < 0. Found ${e[t]} at dim ${t}`);if(r===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);let i=e.slice();return i[r]=t/n,i}function O(e,t){let n=t.length;return e=e==null?t.map((e,t)=>t):[].concat(e),b(e.every(e=>e>=-n&&e<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),b(e.every(e=>T(e)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(e=>e<0?n+e:e)}function ne(e,t){let n=[],r=[],i=t!=null&&Array.isArray(t)&&t.length===0,a=t==null||i?null:O(t,e).sort(),o=0;for(let t=0;t<e.length;++t){if(a!=null){if(a[o]===t&&e[t]!==1)throw Error(`Can't squeeze axis ${t} since its dim '${e[t]}' is not 1`);(a[o]==null||a[o]>t)&&e[t]===1&&(n.push(e[t]),r.push(t)),a[o]<=t&&o++}e[t]!==1&&(n.push(e[t]),r.push(t))}return{newShape:n,keptDims:r}}function re(e,t){return ie(e,t)}function ie(e,t){let n=null;if(e==null||e===`float32`)n=new Float32Array(t);else if(e===`int32`)n=new Int32Array(t);else if(e===`bool`)n=new Uint8Array(t);else if(e===`string`)n=Array(t);else throw Error(`Unknown data type ${e}`);return n}function ae(e,t){for(let n=0;n<e.length;n++){let r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function oe(e){return e===`bool`||e===`complex64`||e===`float32`||e===`int32`||e===`string`}function se(e,t){return!(t===`complex64`||t===`float32`&&e!==`complex64`||t===`int32`&&e!==`float32`&&e!==`complex64`||t===`bool`&&e===`bool`)}function ce(e){if(e===`float32`||e===`int32`)return 4;if(e===`complex64`)return 8;if(e===`bool`)return 1;throw Error(`Unknown dtype ${e}`)}function le(e){if(e==null)return 0;let t=0;return e.forEach(e=>t+=e.length),t}function ue(e){return typeof e==`string`||e instanceof String}function de(e){return typeof e==`boolean`}function fe(e){return typeof e==`number`}function pe(e){return Array.isArray(e)?pe(e[0]):e instanceof Float32Array?`float32`:e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?`int32`:fe(e)?`float32`:ue(e)?`string`:de(e)?`bool`:`float32`}function me(e){return!!(e&&e.constructor&&e.call&&e.apply)}function he(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function k(e){let t=e.length;if(t<2)return[];let n=Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function ge(e,t,n,r=!1){let i=[];if(t.length===1){let a=t[0]*(r?2:1);for(let t=0;t<a;t++)i[t]=n[e+t]}else{let a=t[0],o=t.slice(1),s=o.reduce((e,t)=>e*t)*(r?2:1);for(let t=0;t<a;t++)i[t]=ge(e+t*s,o,n,r)}return i}function _e(e,t,n=!1){if(e.length===0)return t[0];let r=e.reduce((e,t)=>e*t)*(n?2:1);if(r===0)return[];if(r!==t.length)throw Error(`[${e}] does not match the input size ${t.length}${n?` for a complex tensor`:``}.`);return ge(0,e,t,n)}function ve(e,t){if(Array.isArray(e))return e;if(t===`float32`)return e instanceof Float32Array?e:new Float32Array(e);if(t===`int32`)return e instanceof Int32Array?e:new Int32Array(e);if(t===`bool`||t===`string`)return Uint8Array.from(new Int32Array(e));throw Error(`Unknown dtype ${t}`)}function ye(e,t){let n=be(e,t);for(let e=0;e<n.length;e++)n[e]=1;return n}function be(e,t){if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`)return new Uint8Array(e);throw Error(`Unknown data type ${t}`)}function xe(e,t){let n=e.reduce((e,t)=>e*t,1);if(t==null||t===`float32`)return _e(e,new Float32Array(n));if(t===`int32`)return _e(e,new Int32Array(n));if(t===`bool`)return _e(e,new Uint8Array(n));throw Error(`Unknown data type ${t}`)}function Se(e){e.forEach(t=>{b(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function Ce(e,t,n){if(t===0)return 0;if(t===1)return e[0];let r=e[e.length-1];for(let t=0;t<e.length-1;++t)r+=n[t]*e[t];return r}function we(e,t,n){if(t===0)return[];if(t===1)return[e];let r=Array(t);for(let t=0;t<r.length-1;++t)r[t]=Math.floor(e/n[t]),e-=r[t]*n[t];return r[r.length-1]=e,r}function Te(e){return e&&e.then&&typeof e.then==`function`}var Ee=`tfjsflags`,De=class{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Oe,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&(A().getBool(`IS_TEST`)||A().getBool(`PROD`)||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},this.urlFlags[e]!=null){let t=this.urlFlags[e];A().getBool(`IS_TEST`)||A().getBool(`PROD`)||console.warn(`Setting feature override from URL ${e}: ${t}.`),this.set(e,t)}}async getAsync(e){return e in this.flags||(this.flags[e]=await this.evaluateFlag(e)),this.flags[e]}get(e){if(e in this.flags)return this.flags[e];let t=this.evaluateFlag(e);if(Te(t))throw Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(this.global===void 0||this.global.location===void 0||this.global.location.search===void 0)return;let e=this.getQueryParams(this.global.location.search);Ee in e&&e[Ee].split(`,`).forEach(e=>{let[t,n]=e.split(`:`);this.urlFlags[t]=Ae(t,n)})}};function Oe(e){let t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...n)=>(ke(t,n[0],n[1]),n.join(`=`))),t}function ke(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||``)}function Ae(e,t){let n=t.toLowerCase();return n===`true`||n===`false`?n===`true`:`${+n}`===n?+n:t}function A(){return je}var je=null;function Me(e){je=e}var Ne;function Pe(){if(Ne==null){let e;if(typeof window<`u`)e=window;else if(typeof global<`u`)e=global;else if(typeof process<`u`)e=process;else if(typeof self<`u`)e=self;else throw Error(`Could not find a global object`);Ne=e}return Ne}function Fe(){let e=Pe();return e._tfGlobals??=new Map,e._tfGlobals}function Ie(e,t){let n=Fe();if(n.has(e))return n.get(e);{let r=t();return n.set(e,r),n.get(e)}}var Le=`Acos`,Re=`Acosh`,ze=`AddN`,Be=`ArgMax`,Ve=`ArgMin`,He=`Asin`,Ue=`Asinh`,We=`Atan`,Ge=`Atanh`,Ke=`Atan2`,qe=`AvgPool`,Je=`AvgPoolGrad`,Ye=`AvgPool3D`,Xe=`AvgPool3DGrad`,Ze=`BatchMatMul`,Qe=`BatchToSpaceND`,$e=`Bincount`,et=`BitwiseAnd`,tt=`BroadcastTo`,nt=`BroadcastArgs`,rt=`Cast`,it=`Ceil`,at=`ClipByValue`,ot=`Complex`,st=`ComplexAbs`,ct=`Concat`,lt=`Conv2D`,ut=`Conv2DBackpropFilter`,dt=`Conv2DBackpropInput`,ft=`Conv3D`,pt=`Conv3DBackpropFilterV2`,mt=`Conv3DBackpropInputV2`,ht=`Cosh`,gt=`Cumprod`,_t=`Cumsum`,vt=`CropAndResize`,yt=`DenseBincount`,bt=`DepthToSpace`,xt=`DepthwiseConv2dNative`,St=`DepthwiseConv2dNativeBackpropFilter`,Ct=`DepthwiseConv2dNativeBackpropInput`,wt=`Diag`,Tt=`Dilation2D`,Et=`Dilation2DBackpropInput`,Dt=`Dilation2DBackpropFilter`,Ot=`Draw`,kt=`RealDiv`,At=`Einsum`,jt=`EluGrad`,Mt=`Equal`,Nt=`ExpandDims`,Pt=`Expm1`,Ft=`Fill`,It=`FlipLeftRight`,Lt=`Floor`,Rt=`FloorDiv`,zt=`FusedBatchNorm`,Bt=`GatherV2`,Vt=`GatherNd`,Ht=`Greater`,Ut=`GreaterEqual`,Wt=`Identity`,Gt=`IFFT`,Kt=`Imag`,qt=`IsFinite`,Jt=`IsInf`,Yt=`IsNan`,Xt=`LeakyRelu`,Zt=`Less`,Qt=`LessEqual`,$t=`LinSpace`,en=`Log1p`,tn=`LogicalAnd`,nn=`LogicalNot`,rn=`LogicalOr`,an=`LogSoftmax`,on=`LRNGrad`,sn=`Maximum`,cn=`MaxPool`,ln=`MaxPoolGrad`,un=`MaxPool3D`,dn=`MaxPool3DGrad`,fn=`MaxPoolWithArgmax`,pn=`Mean`,mn=`Minimum`,hn=`MirrorPad`,gn=`Multinomial`,_n=`Multiply`,vn=`NotEqual`,yn=`NonMaxSuppressionV3`,bn=`NonMaxSuppressionV4`,xn=`NonMaxSuppressionV5`,Sn=`OnesLike`,Cn=`OneHot`,wn=`Pack`,Tn=`PadV2`,En=`Prelu`,Dn=`Prod`,On=`RaggedGather`,kn=`RaggedRange`,An=`RaggedTensorToTensor`,jn=`Range`,Mn=`Real`,Nn=`Reciprocal`,Pn=`Relu`,Fn=`Reshape`,In=`ResizeNearestNeighbor`,Ln=`ResizeNearestNeighborGrad`,Rn=`ResizeBilinear`,zn=`ResizeBilinearGrad`,Bn=`Relu6`,Vn=`Reverse`,Hn=`Round`,Un=`Rsqrt`,Wn=`ScatterNd`,Gn=`TensorScatterUpdate`,Kn=`SearchSorted`,qn=`Select`,Jn=`Selu`,Yn=`Slice`,Xn=`Sinh`,Zn=`Sign`,Qn=`Sigmoid`,$n=`Softplus`,er=`Sqrt`,tr=`SpaceToBatchND`,nr=`SplitV`,rr=`Softmax`,ir=`SparseFillEmptyRows`,ar=`SparseReshape`,or=`SparseSegmentMean`,sr=`SparseSegmentSum`,cr=`SparseToDense`,lr=`SquaredDifference`,ur=`Square`,dr=`StaticRegexReplace`,fr=`StridedSlice`,pr=`StringNGrams`,mr=`StringSplit`,hr=`StringToHashBucketFast`,gr=`Tanh`,_r=`Tile`,vr=`TopK`,yr=`Transform`,br=`Transpose`,xr=`Unique`,Sr=`Unpack`,Cr=`UnsortedSegmentSum`,wr=`ZerosLike`,Tr=`Step`,Er=`FromPixels`,Dr=`RotateWithOffset`,Or=`_FusedMatMul`,kr=`FusedConv2D`,Ar=`FusedDepthwiseConv2D`;function jr(...e){A().getBool(`IS_TEST`)||A().getBool(`PROD`)||console.warn(...e)}function Mr(...e){A().getBool(`IS_TEST`)||A().getBool(`PROD`)||console.log(...e)}var Nr=Ie(`kernelRegistry`,()=>new Map),Pr=Ie(`gradRegistry`,()=>new Map);function Fr(e,t){let n=Br(e,t);return Nr.get(n)}function Ir(e){return Pr.get(e)}function Lr(e){let t=Nr.entries(),n=[];for(;;){let{done:r,value:i}=t.next();if(r)break;let[a,o]=i,[s]=a.split(`_`);s===e&&n.push(o)}return n}function Rr(e){let{kernelName:t,backendName:n}=e,r=Br(t,n);Nr.has(r)&&jr(`The kernel '${t}' for backend '${n}' is already registered`),Nr.set(r,e)}function zr(e){let{kernelName:t}=e;Pr.has(t)&&A().getBool(`DEBUG`)&&jr(`Overriding the gradient for '${t}'`),Pr.set(t,e)}function Br(e,t){return`${t}_${e}`}function Vr(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var Hr=l(o(((e,t)=>{t.exports=r;var n=null;try{n=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function r(e,t,n){this.low=e|0,this.high=t|0,this.unsigned=!!n}r.prototype.__isLong__,Object.defineProperty(r.prototype,"__isLong__",{value:!0});function i(e){return(e&&e.__isLong__)===!0}r.isLong=i;var a={},o={};function s(e,t){var n,r,i;return t?(e>>>=0,(i=0<=e&&e<256)&&(r=o[e],r)?r:(n=l(e,(e|0)<0?-1:0,!0),i&&(o[e]=n),n)):(e|=0,(i=-128<=e&&e<128)&&(r=a[e],r)?r:(n=l(e,e<0?-1:0,!1),i&&(a[e]=n),n))}r.fromInt=s;function c(e,t){if(isNaN(e))return t?b:y;if(t){if(e<0)return b;if(e>=g)return T}else{if(e<=-_)return E;if(e+1>=_)return w}return e<0?c(-e,t).neg():l(e%h|0,e/h|0,t)}r.fromNumber=c;function l(e,t,n){return new r(e,t,n)}r.fromBits=l;var u=Math.pow;function d(e,t,n){if(e.length===0)throw Error(`empty string`);if(e===`NaN`||e===`Infinity`||e===`+Infinity`||e===`-Infinity`)return y;if(typeof t==`number`?(n=t,t=!1):t=!!t,n||=10,n<2||36<n)throw RangeError(`radix`);var r;if((r=e.indexOf(`-`))>0)throw Error(`interior hyphen`);if(r===0)return d(e.substring(1),t,n).neg();for(var i=c(u(n,8)),a=y,o=0;o<e.length;o+=8){var s=Math.min(8,e.length-o),l=parseInt(e.substring(o,o+s),n);if(s<8){var f=c(u(n,s));a=a.mul(f).add(c(l))}else a=a.mul(i),a=a.add(c(l))}return a.unsigned=t,a}r.fromString=d;function f(e,t){return typeof e==`number`?c(e,t):typeof e==`string`?d(e,t):l(e.low,e.high,typeof t==`boolean`?t:e.unsigned)}r.fromValue=f;var p=65536,m=1<<24,h=p*p,g=h*h,_=g/2,v=s(m),y=s(0);r.ZERO=y;var b=s(0,!0);r.UZERO=b;var x=s(1);r.ONE=x;var S=s(1,!0);r.UONE=S;var C=s(-1);r.NEG_ONE=C;var w=l(-1,2147483647,!1);r.MAX_VALUE=w;var T=l(-1,-1,!0);r.MAX_UNSIGNED_VALUE=T;var E=l(0,-2147483648,!1);r.MIN_VALUE=E;var D=r.prototype;D.toInt=function(){return this.unsigned?this.low>>>0:this.low},D.toNumber=function(){return this.unsigned?(this.high>>>0)*h+(this.low>>>0):this.high*h+(this.low>>>0)},D.toString=function(e){if(e||=10,e<2||36<e)throw RangeError(`radix`);if(this.isZero())return`0`;if(this.isNegative()){if(this.eq(E)){var t=c(e),n=this.div(t),r=n.mul(t).sub(this);return n.toString(e)+r.toInt().toString(e)}return`-`+this.neg().toString(e)}for(var i=c(u(e,6),this.unsigned),a=this,o=``;;){var s=a.div(i),l=(a.sub(s.mul(i)).toInt()>>>0).toString(e);if(a=s,a.isZero())return l+o;for(;l.length<6;)l=`0`+l;o=``+l+o}},D.getHighBits=function(){return this.high},D.getHighBitsUnsigned=function(){return this.high>>>0},D.getLowBits=function(){return this.low},D.getLowBitsUnsigned=function(){return this.low>>>0},D.getNumBitsAbs=function(){if(this.isNegative())return this.eq(E)?64:this.neg().getNumBitsAbs();for(var e=this.high==0?this.low:this.high,t=31;t>0&&!(e&1<<t);t--);return this.high==0?t+1:t+33},D.isZero=function(){return this.high===0&&this.low===0},D.eqz=D.isZero,D.isNegative=function(){return!this.unsigned&&this.high<0},D.isPositive=function(){return this.unsigned||this.high>=0},D.isOdd=function(){return(this.low&1)==1},D.isEven=function(){return!(this.low&1)},D.equals=function(e){return i(e)||(e=f(e)),this.unsigned!==e.unsigned&&this.high>>>31==1&&e.high>>>31==1?!1:this.high===e.high&&this.low===e.low},D.eq=D.equals,D.notEquals=function(e){return!this.eq(e)},D.neq=D.notEquals,D.ne=D.notEquals,D.lessThan=function(e){return this.comp(e)<0},D.lt=D.lessThan,D.lessThanOrEqual=function(e){return this.comp(e)<=0},D.lte=D.lessThanOrEqual,D.le=D.lessThanOrEqual,D.greaterThan=function(e){return this.comp(e)>0},D.gt=D.greaterThan,D.greaterThanOrEqual=function(e){return this.comp(e)>=0},D.gte=D.greaterThanOrEqual,D.ge=D.greaterThanOrEqual,D.compare=function(e){if(i(e)||(e=f(e)),this.eq(e))return 0;var t=this.isNegative(),n=e.isNegative();return t&&!n?-1:!t&&n?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1},D.comp=D.compare,D.negate=function(){return!this.unsigned&&this.eq(E)?E:this.not().add(x)},D.neg=D.negate,D.add=function(e){i(e)||(e=f(e));var t=this.high>>>16,n=this.high&65535,r=this.low>>>16,a=this.low&65535,o=e.high>>>16,s=e.high&65535,c=e.low>>>16,u=e.low&65535,d=0,p=0,m=0,h=0;return h+=a+u,m+=h>>>16,h&=65535,m+=r+c,p+=m>>>16,m&=65535,p+=n+s,d+=p>>>16,p&=65535,d+=t+o,d&=65535,l(m<<16|h,d<<16|p,this.unsigned)},D.subtract=function(e){return i(e)||(e=f(e)),this.add(e.neg())},D.sub=D.subtract,D.multiply=function(e){if(this.isZero())return y;if(i(e)||(e=f(e)),n)return l(n.mul(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(e.isZero())return y;if(this.eq(E))return e.isOdd()?E:y;if(e.eq(E))return this.isOdd()?E:y;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(v)&&e.lt(v))return c(this.toNumber()*e.toNumber(),this.unsigned);var t=this.high>>>16,r=this.high&65535,a=this.low>>>16,o=this.low&65535,s=e.high>>>16,u=e.high&65535,d=e.low>>>16,p=e.low&65535,m=0,h=0,g=0,_=0;return _+=o*p,g+=_>>>16,_&=65535,g+=a*p,h+=g>>>16,g&=65535,g+=o*d,h+=g>>>16,g&=65535,h+=r*p,m+=h>>>16,h&=65535,h+=a*d,m+=h>>>16,h&=65535,h+=o*u,m+=h>>>16,h&=65535,m+=t*p+r*d+a*u+o*s,m&=65535,l(g<<16|_,m<<16|h,this.unsigned)},D.mul=D.multiply,D.divide=function(e){if(i(e)||(e=f(e)),e.isZero())throw Error(`division by zero`);if(n)return!this.unsigned&&this.high===-2147483648&&e.low===-1&&e.high===-1?this:l((this.unsigned?n.div_u:n.div_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned);if(this.isZero())return this.unsigned?b:y;var t,r,a;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return b;if(e.gt(this.shru(1)))return S;a=b}else{if(this.eq(E))return e.eq(x)||e.eq(C)?E:e.eq(E)?x:(t=this.shr(1).div(e).shl(1),t.eq(y)?e.isNegative()?x:C:(r=this.sub(e.mul(t)),a=t.add(r.div(e)),a));if(e.eq(E))return this.unsigned?b:y;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();a=y}for(r=this;r.gte(e);){t=Math.max(1,Math.floor(r.toNumber()/e.toNumber()));for(var o=Math.ceil(Math.log(t)/Math.LN2),s=o<=48?1:u(2,o-48),d=c(t),p=d.mul(e);p.isNegative()||p.gt(r);)t-=s,d=c(t,this.unsigned),p=d.mul(e);d.isZero()&&(d=x),a=a.add(d),r=r.sub(p)}return a},D.div=D.divide,D.modulo=function(e){return i(e)||(e=f(e)),n?l((this.unsigned?n.rem_u:n.rem_s)(this.low,this.high,e.low,e.high),n.get_high(),this.unsigned):this.sub(this.div(e).mul(e))},D.mod=D.modulo,D.rem=D.modulo,D.not=function(){return l(~this.low,~this.high,this.unsigned)},D.and=function(e){return i(e)||(e=f(e)),l(this.low&e.low,this.high&e.high,this.unsigned)},D.or=function(e){return i(e)||(e=f(e)),l(this.low|e.low,this.high|e.high,this.unsigned)},D.xor=function(e){return i(e)||(e=f(e)),l(this.low^e.low,this.high^e.high,this.unsigned)},D.shiftLeft=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):l(0,this.low<<e-32,this.unsigned)},D.shl=D.shiftLeft,D.shiftRight=function(e){return i(e)&&(e=e.toInt()),(e&=63)==0?this:e<32?l(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):l(this.high>>e-32,this.high>=0?0:-1,this.unsigned)},D.shr=D.shiftRight,D.shiftRightUnsigned=function(e){if(i(e)&&(e=e.toInt()),e&=63,e===0)return this;var t=this.high;if(e<32){var n=this.low;return l(n>>>e|t<<32-e,t>>>e,this.unsigned)}return l(e===32?t:t>>>e-32,0,this.unsigned)},D.shru=D.shiftRightUnsigned,D.shr_u=D.shiftRightUnsigned,D.toSigned=function(){return this.unsigned?l(this.low,this.high,!1):this},D.toUnsigned=function(){return this.unsigned?this:l(this.low,this.high,!0)},D.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()},D.toBytesLE=function(){var e=this.high,t=this.low;return[t&255,t>>>8&255,t>>>16&255,t>>>24,e&255,e>>>8&255,e>>>16&255,e>>>24]},D.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,e&255,t>>>24,t>>>16&255,t>>>8&255,t&255]},r.fromBytes=function(e,t,n){return n?r.fromBytesLE(e,t):r.fromBytesBE(e,t)},r.fromBytesLE=function(e,t){return new r(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)},r.fromBytesBE=function(e,t){return new r(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)}}))()),Ur=Hr.default||Hr;function Wr(e){return Ur.fromString(e,!0,16)}var Gr=Wr(`c3a5c85c97cb3127`),Kr=Wr(`b492b66fbe98f273`),qr=Wr(`9ae16a3b2f90404f`);function Jr(e){return e.xor(e.shru(47))}function Yr(e,t,n){let r=e.slice(t,t+n);return Ur.fromBytes(Array.from(r),!0,!0)}function Xr(e,t){return Yr(e,t,8)}function Zr(e,t){return Yr(e,t,4)}function Qr(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function $r(e,t,n=Wr(`9ddfea08eb382d69`)){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let i=t.xor(r).mul(n);return i=i.xor(i.shru(47)),i=i.mul(n),i}function ei(e,t,n,r,i,a){i=i.add(e),a=Qr(a.add(i).add(r),21);let o=i;return i=i.add(t),i=i.add(n),a=a.add(Qr(i,44)),[i.add(r),a.add(o)]}function ti(e,t,n,r){return ei(Xr(e,t),Xr(e,t+8),Xr(e,t+16),Xr(e,t+24),n,r)}function ni(e,t=e.length){if(t>=8){let n=qr.add(t*2),r=Xr(e,0).add(qr),i=Xr(e,t-8);return $r(Qr(i,37).mul(n).add(r),Qr(r,25).add(i).mul(n),n)}if(t>=4){let n=qr.add(t*2);return $r(Zr(e,0).shl(3).add(t),Zr(e,t-4),n)}if(t>0){let n=e[0],r=e[t>>1],i=e[t-1],a=n+(r<<8),o=t+(i<<2);return Jr(qr.mul(a).xor(Gr.mul(o))).mul(qr)}return qr}function ri(e,t=e.length){let n=qr.add(t*2),r=Xr(e,0).mul(Kr),i=Xr(e,8),a=Xr(e,t-8).mul(n),o=Xr(e,t-16).mul(qr);return $r(Qr(r.add(i),43).add(Qr(a,30)).add(o),r.add(Qr(i.add(qr),18)).add(a),n)}function ii(e,t=e.length){let n=qr.add(t*2),r=Xr(e,0).mul(qr),i=Xr(e,8),a=Xr(e,t-8).mul(n),o=Xr(e,t-16).mul(qr),s=Qr(r.add(i),43).add(Qr(a,30)).add(o),c=$r(s,r.add(Qr(i.add(qr),18)).add(a),n),l=Xr(e,16).mul(n),u=Xr(e,24),d=s.add(Xr(e,t-32)).mul(n),f=c.add(Xr(e,t-24)).mul(n);return $r(Qr(l.add(u),43).add(Qr(d,30)).add(f),l.add(Qr(u.add(r),18)).add(d),n)}function ai(e,t=e.length){let n=Ur.fromNumber(81,!0);if(t<=32)return t<=16?ni(e,t):ri(e,t);if(t<=64)return ii(e,t);let r=n,i=n.mul(Kr).add(113),a=Jr(i.mul(qr).add(113)).mul(qr),o=[Ur.UZERO,Ur.UZERO],s=[Ur.UZERO,Ur.UZERO];r=r.mul(qr).add(Xr(e,0));let c=0,l=(t-1>>6)*64,u=l+(t-1&63)-63;do r=Qr(r.add(i).add(o[0]).add(Xr(e,c+8)),37).mul(Kr),i=Qr(i.add(o[1]).add(Xr(e,c+48)),42).mul(Kr),r=r.xor(s[1]),i=i.add(o[0]).add(Xr(e,c+40)),a=Qr(a.add(s[0]),33).mul(Kr),o=ti(e,c,o[1].mul(Kr),r.add(s[0])),s=ti(e,c+32,a.add(s[1]),i.add(Xr(e,c+16))),[a,r]=[r,a],c+=64;while(c!==l);let d=Kr.add(a.and(255).shl(1));return c=u,s[0]=s[0].add(t-1&63),o[0]=o[0].add(s[0]),s[0]=s[0].add(o[0]),r=Qr(r.add(i).add(o[0]).add(Xr(e,c+8)),37).mul(d),i=Qr(i.add(o[1]).add(Xr(e,c+48)),42).mul(d),r=r.xor(s[1].mul(9)),i=i.add(o[0].mul(9).add(Xr(e,c+40))),a=Qr(a.add(s[0]),33).mul(d),o=ti(e,c,o[1].mul(d),r.add(s[0])),s=ti(e,c+32,a.add(s[1]),i.add(Xr(e,c+16))),[a,r]=[r,a],$r($r(o[0],s[0],d).add(Jr(i).mul(Gr)).add(a),$r(o[1],s[1],d).add(r),d)}function oi(e,t){return t===`string`?ui(e):ci([e],t)}function si(e,t){return e instanceof Float32Array&&t===`float32`||e instanceof Int32Array&&t===`int32`||e instanceof Uint8Array&&t===`bool`}function ci(e,t){if(t===`string`)throw Error(`Cannot convert a string[] to a TypedArray`);if(Array.isArray(e)&&(e=pi(e)),A().getBool(`DEBUG`)&&ae(e,t),si(e,t))return e;if(t==null||t===`float32`||t===`complex64`)return new Float32Array(e);if(t===`int32`)return new Int32Array(e);if(t===`bool`){let t=new Uint8Array(e.length);for(let n=0;n<t.length;++n)Math.round(e[n])!==0&&(t[n]=1);return t}throw Error(`Unknown data type ${t}`)}function li(){return A().platform.now()}function ui(e,t=`utf-8`){return t||=`utf-8`,A().platform.encode(e,t)}function di(e,t=`utf-8`){return t||=`utf-8`,A().platform.decode(e,t)}function fi(e){return A().platform.isTypedArray==null?Vr(e):A().platform.isTypedArray(e)}function pi(e,t=[],n=!1){if(t??=[],typeof e==`boolean`||typeof e==`number`||typeof e==`string`||Te(e)||e==null||fi(e)&&n)t.push(e);else if(Array.isArray(e)||fi(e))for(let r=0;r<e.length;++r)pi(e[r],t,n);else{let r=-1;for(let t of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(t)&&(r=Math.max(r,Number(t)));for(let i=0;i<=r;i++)pi(e[i],t,n)}return t}var mi=class{constructor(e,t){this.backendTimer=e,this.logger=t,t??(this.logger=new gi)}profileKernel(e,t,n){let r,i=()=>{r=n()},a,o=li();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(i);else{i();for(let e of r)e.dataSync();a=Promise.resolve({kernelMs:li()-o})}if(A().getBool(`CHECK_COMPUTATION_FOR_ERRORS`))for(let t=0;t<r.length;t++){let n=r[t];n.data().then(t=>{hi(t,n.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:a.then(e=>e.kernelMs),extraInfo:a.then(e=>e.getExtraProfileInfo==null?``:e.getExtraProfileInfo())}}logKernelProfile(e){let{kernelName:t,outputs:n,timeMs:r,inputs:i,extraInfo:a}=e;n.forEach(e=>{Promise.all([e.data(),r,a]).then(n=>{this.logger.logKernelProfile(t,e,n[0],n[1],i,n[2])})})}};function hi(e,t,n){if(t!==`float32`)return!1;for(let t=0;t<e.length;t++){let r=e[t];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}var gi=class{logKernelProfile(e,t,n,r,i,a){let o=typeof r==`number`?D(`${r}ms`,9):r.error,s=D(e,25),c=t.rank,l=t.size,u=D(t.shape.toString(),14),d=``;for(let e in i){let n=i[e];if(n!=null){let r=n.shape||t.shape,i=r.length;d+=`${e}: ${i}D ${i>0?r:``} `}}console.log(`%c${s}\t%c${o}\t%c${c}D ${u}\t%c${l}\t%c${d}\t%c${a}`,`font-weight:bold`,`color:red`,`color:blue`,`color: orange`,`color: green`,`color: steelblue`)}};function _i(e,t,n){let r={},i={};for(let e=0;e<t.length;e++)r[t[e].id]=!0;for(let n=0;n<e.length;n++){let a=e[n],o=a.inputs;for(let e in o){let n=o[e],s=!1;for(let e=0;e<t.length;e++)if(r[n.id]){a.outputs.forEach(e=>r[e.id]=!0),s=!0,i[a.id]=!0;break}if(s)break}}let a={};a[n.id]=!0;let o={};for(let t=e.length-1;t>=0;t--){let n=e[t],r=n.inputs;for(let e=0;e<n.outputs.length;e++)if(a[n.outputs[e].id]){for(let e in r)a[r[e].id]=!0,o[n.id]=!0;break}}let s=[];for(let t=0;t<e.length;t++){let n=e[t];if(i[n.id]&&o[n.id]){let e={};for(let t in n.inputs){let i=n.inputs[t];r[i.id]&&(e[t]=i)}let t=Object.assign({},n);t.inputs=e,t.outputs=n.outputs,s.push(t)}}return s}function vi(e,t,n,r){for(let i=t.length-1;i>=0;i--){let a=t[i],o=[];if(a.outputs.forEach(t=>{let n=e[t.id];n==null?o.push(null):o.push(n)}),a.gradient==null)throw Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);let s=a.gradient(o);for(let t in a.inputs){if(!(t in s))throw Error(`Cannot backprop through input ${t}. Available gradients found: ${Object.keys(s)}.`);let i=n(()=>s[t]());if(i.dtype!==`float32`)throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${t} must have 'float32' dtype, but has '${i.dtype}'`);let o=a.inputs[t];if(!w(i.shape,o.shape))throw Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${t}' has shape '${i.shape}', which does not match the shape of the input '${o.shape}'`);if(e[o.id]==null)e[o.id]=i;else{let t=e[o.id];e[o.id]=r(t,i),t.dispose()}}}}var yi=20,bi=3,xi=7;function Si(e,t,n,r){let i=k(t),a=Ci(e,t,n,i),o=t.length,s=Ei(e,t,n,i,a),c=[`Tensor`];return r&&(c.push(`  dtype: ${n}`),c.push(`  rank: ${o}`),c.push(`  shape: [${t}]`),c.push(`  values:`)),c.push(s.map(e=>`    `+e).join(`
`)),c.join(`
`)}function Ci(e,t,n,r){let i=C(t),a=r[r.length-1],o=Array(a).fill(0),s=t.length,c=n===`complex64`?Di(e):e;if(s>1)for(let e=0;e<i/a;e++){let t=e*a;for(let e=0;e<a;e++)o[e]=Math.max(o[e],wi(c[t+e],0,n).length)}return o}function wi(e,t,n){let r;return r=Array.isArray(e)?`${parseFloat(e[0].toFixed(xi))} + ${parseFloat(e[1].toFixed(xi))}j`:ue(e)?`'${e}'`:n===`bool`?Ti(e):parseFloat(e.toFixed(xi)).toString(),D(r,t)}function Ti(e){return e===0?`false`:`true`}function Ei(e,t,n,r,i,a=!0){let o=n===`complex64`?2:1,s=t[0],c=t.length;if(c===0)return n===`complex64`?[wi(Di(e)[0],0,n)]:n===`bool`?[Ti(e[0])]:[e[0].toString()];if(c===1){if(s>yi){let t=bi*o,r=Array.from(e.slice(0,t)),a=Array.from(e.slice((s-bi)*o,s*o));return n===`complex64`&&(r=Di(r),a=Di(a)),[`[`+r.map((e,t)=>wi(e,i[t],n)).join(`, `)+`, ..., `+a.map((e,t)=>wi(e,i[s-bi+t],n)).join(`, `)+`]`]}return[`[`+(n===`complex64`?Di(e):Array.from(e)).map((e,t)=>wi(e,i[t],n)).join(`, `)+`]`]}let l=t.slice(1),u=r.slice(1),d=r[0]*o,f=[];if(s>yi){for(let t=0;t<bi;t++){let r=t*d,a=r+d;f.push(...Ei(e.slice(r,a),l,n,u,i,!1))}f.push(`...`);for(let t=s-bi;t<s;t++){let r=t*d,a=r+d;f.push(...Ei(e.slice(r,a),l,n,u,i,t===s-1))}}else for(let t=0;t<s;t++){let r=t*d,a=r+d;f.push(...Ei(e.slice(r,a),l,n,u,i,t===s-1))}let p=c===2?`,`:``;f[0]=`[`+(s>0?f[0]+p:``);for(let e=1;e<f.length-1;e++)f[e]=` `+f[e]+p;let m=`,
`;for(let e=2;e<c;e++)m+=`
`;return f[f.length-1]=` `+f[f.length-1]+`]`+(a?``:m),f}function Di(e){let t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}var Oi=class{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=C(e),n!=null){let e=n.length;b(e===this.size,()=>`Length of values '${e}' does not match the size inferred by the shape '${this.size}'.`)}if(t===`complex64`)throw Error(`complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).`);this.values=n||ie(t,this.size),this.strides=k(e)}set(e,...t){t.length===0&&(t=[0]),b(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);let n=this.locToIndex(t);this.values[n]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(let n of e){if(n<0||n>=this.shape[t]){let t=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw Error(t)}t++}let n=e[e.length-1];for(let t=0;t<e.length-1;++t)n+=this.strides[t]*e[t];return this.values[n]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];let t=Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return ki().makeTensor(this.values,this.shape,this.dtype)}},ki=null,Ai=null;function ji(e){ki=e}function Mi(e){Ai=e}var Ni=class{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||`float32`,this.size=C(e),this.strides=k(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():`higher`}get rank(){return this.shape.length}async buffer(){let e=await this.data();return Ai.buffer(this.shape,this.dtype,e)}bufferSync(){return Ai.buffer(this.shape,this.dtype,this.dataSync())}async array(){let e=await this.data();return _e(this.shape,e,this.dtype===`complex64`)}arraySync(){return _e(this.shape,this.dataSync(),this.dtype===`complex64`)}async data(){this.throwIfDisposed();let e=ki().read(this.dataId);if(this.dtype===`string`){let t=await e;try{return t.map(e=>di(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}}return e}dataToGPU(e){return this.throwIfDisposed(),ki().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();let e=ki().readSync(this.dataId);if(this.dtype===`string`)try{return e.map(e=>di(e))}catch{throw Error(`Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().`)}return e}async bytes(){this.throwIfDisposed();let e=await ki().read(this.dataId);return this.dtype===`string`?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),ki().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw Error(`Tensor is disposed.`)}print(e=!1){return Ai.print(this,e)}clone(){return this.throwIfDisposed(),Ai.clone(this)}toString(e=!1){return Si(this.dataSync(),this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Ai.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),ki().makeVariable(this,e,t,n)}};Object.defineProperty(Ni,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function j(){return Ie(`Tensor`,()=>Ni)}j();var Pi=class extends Ni{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!w(e.shape,this.shape))throw Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);ki().disposeTensor(this),this.dataId=e.dataId,ki().incRef(this,null)}dispose(){ki().disposeVariable(this),this.isDisposedInternal=!0}};Object.defineProperty(Pi,Symbol.hasInstance,{value:e=>e instanceof Ni&&e.assign!=null&&e.assign instanceof Function});var Fi;(function(e){e.R0=`R0`,e.R1=`R1`,e.R2=`R2`,e.R3=`R3`,e.R4=`R4`,e.R5=`R5`,e.R6=`R6`})(Fi||={});var Ii;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`int32`,e.complex64=`complex64`})(Ii||={});var Li;(function(e){e.float32=`float32`,e.int32=`int32`,e.bool=`bool`,e.complex64=`complex64`})(Li||={});var Ri;(function(e){e.float32=`float32`,e.int32=`float32`,e.bool=`float32`,e.complex64=`complex64`})(Ri||={});var zi;(function(e){e.float32=`complex64`,e.int32=`complex64`,e.bool=`complex64`,e.complex64=`complex64`})(zi||={});var Bi={float32:Ri,int32:Ii,bool:Li,complex64:zi};function Vi(e,t){if(e===`string`||t===`string`){if(e===`string`&&t===`string`)return`string`;throw Error(`Can not upcast ${e} with ${t}`)}return Bi[e][t]}function Hi(e){return Vi(e,`int32`)}function Ui(e){return typeof e==`object`&&!!e&&`texture`in e&&e.texture instanceof WebGLTexture}function Wi(e){return typeof GPUBuffer<`u`&&typeof e==`object`&&!!e&&`buffer`in e&&e.buffer instanceof GPUBuffer}function Gi(e,t){if(e.dtype===t.dtype)return[e,t];let n=Vi(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function Ki(e,t){return t.some(t=>t.id===e.id)}function qi(e){let t=[];return Ji(e,t,new Set),t}function Ji(e,t,n){if(e==null)return;if(e instanceof Ni){t.push(e);return}if(!Yi(e))return;let r=e;for(let e in r){let i=r[e];n.has(i)||(n.add(i),Ji(i,t,n))}}function Yi(e){return Array.isArray(e)||typeof e==`object`}function Xi(e){return e.kernelName!=null}var Zi=class{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(let e in this.registeredVariables)this.registeredVariables[e].dispose()}},Qi=class e{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Zi}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t];if(await this.initializeBackend(n).success){await this.setBackend(n);return}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}get backend(){if(this.pendingBackendInit!=null)throw Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){let{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry)){if(e in this.registryFactory){let{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null}return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(jr(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;let{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new mi(this.backendInstance),!0}setupRegisteredKernels(){Lr(this.backendName).forEach(e=>{e.setupFunc!=null&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){Lr(e).forEach(t=>{t.disposeFunc!=null&&t.disposeFunc(this.registry[e])})}initializeBackend(e){let t=this.registryFactory[e];if(t==null)throw Error(`Cannot initialize backend ${e}, no registration found.`);try{let n=t.factory();if(n&&!(n instanceof p)&&typeof n.then==`function`){let t=++this.pendingBackendInitId,r=n.then(n=>t<this.pendingBackendInitId?!1:(this.registry[e]=n,this.pendingBackendInit=null,!0)).catch(n=>t<this.pendingBackendInitId?!1:(this.pendingBackendInit=null,jr(`Initialization of backend ${e} failed`),jr(n.stack||n.message),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}return this.registry[e]=n,{success:!0,asyncInit:!1}}catch(t){return jr(`Initialization of backend ${e} failed`),jr(t.stack||t.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw Error(`No backend found in registry.`);return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){let e=this.getSortedBackends();for(let t=0;t<e.length;t++){let n=e[t],{success:r,asyncInit:i}=this.initializeBackend(n);if(i||r)return{name:n,asyncInit:i}}throw Error(`Could not initialize any backends, all backend initializations failed.`)}moveData(e,t){let n=this.state.tensorInfo.get(t),r=n.backend,i=this.readSync(t),a=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,i,n.shape,n.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n=null;if(t==null){if(typeof e!=`function`)throw Error(`Please provide a function to tidy()`);t=e}else{if(typeof e!=`string`&&!(e instanceof String))throw Error(`When calling with two arguments, the first argument to tidy() must be a string`);if(typeof t!=`function`)throw Error(`When calling with two arguments, the 2nd argument to tidy() must be a function`);n=e}let r;return this.scopedRun(()=>this.startScope(n),()=>this.endScope(r),()=>(r=t(),r instanceof Promise&&console.error(`Cannot return a Promise inside of tidy.`),r))}scopedRun(e,t,n){e();try{let e=n();return t(),e}catch(e){throw t(),e}}nextTensorId(){return e.nextTensorId++}nextVariableId(){return e.nextVariableId++}clone(e){let t=M.runKernel(Wt,{x:e}),n={x:e};return this.addTapeNode(this.state.activeScope.name,n,[t],e=>({x:()=>{let t={x:e};return M.runKernel(rt,t,{dtype:`float32`})}}),[],{}),t}runKernel(e,t,n){if(this.backendName??this.backend,Fr(e,this.backendName)==null)throw Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool(`IS_TEST`)}checkKernelForMemLeak(e,t,n){let r=this.backend.numDataIds(),i=0;n.forEach(e=>{i+=e.dtype===`complex64`?3:1});let a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],o=r-t-i-a;if(o>0)throw Error(`Backend '${this.backendName}' has an internal memory leak (${o} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[],r=this.isTapeOn(),i=this.state.numBytes,a=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let o;this.backendName??this.backend;let s,c=Xi(e)?e.kernelName:this.state.activeScope==null?``:this.state.activeScope.name;if(Xi(e)){let{kernelName:t,inputs:i,attrs:a}=e;this.backendName??this.backend;let c=Fr(t,this.backendName);b(c!=null,()=>`Cannot find registered kernel '${t}' for backend '${this.backendName}'`),o=()=>{let e=this.backend.numDataIds();s=c.kernelFunc({inputs:i,attrs:a,backend:this.backend});let o=Array.isArray(s)?s:[s];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(t,e,o);let l=o.map(e=>e.rank==null?this.makeTensorFromTensorInfo(e):e);if(r){let e=this.getTensorsForGradient(t,i,l);n=this.saveTensorsForBackwardMode(e)}return l}}else{let{forwardFunc:t}=e,i=e=>{r&&(n=e.map(e=>this.keep(this.clone(e))))};o=()=>{let e=this.backend.numDataIds();s=this.tidy(()=>t(this.backend,i));let n=Array.isArray(s)?s:[s];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,e,n),n}}let{inputs:l,attrs:u}=e,d=Xi(e)?null:e.backwardsFunc,f;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool(`DEBUG`)&&!this.state.profiling?t=o():(f=this.profiler.profileKernel(c,l,()=>o()),this.ENV.getBool(`DEBUG`)&&this.profiler.logKernelProfile(f),t=f.outputs)}),r&&this.addTapeNode(c,l,t,d,n,u),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-i,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(l).map(e=>l[e]==null?null:l[e].shape),outputShapes:t.map(e=>e.shape),kernelTimeMs:f.timeMs,extraInfo:f.extraInfo}),Array.isArray(s)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(e=>this.keep(this.clone(e)))}getTensorsForGradient(e,t,n){let r=Ir(e);if(r!=null){let e=r.inputsToSave||[],i=r.outputsToSave||[],a;r.saveAllInputs?(b(Array.isArray(t),()=>`saveAllInputs is true, expected inputs to be an array.`),a=Object.keys(t).map(e=>t[e])):a=e.map(e=>t[e]);let o=n.filter((e,t)=>i[t]);return a.concat(o)}return[]}makeTensor(e,t,n,r){if(e==null)throw Error(`Values passed to engine.makeTensor() are null`);n||=`float32`,r||=this.backend;let i=e;n===`string`&&ue(e[0])&&(i=e.map(e=>ui(e)));let a=r.write(i,t,n),o=new Ni(t,n,a,this.nextTensorId());if(this.trackTensor(o,r),n===`string`){let e=this.state.tensorInfo.get(a),t=le(i);this.state.numBytes+=t-e.bytes,e.bytes=t}return o}makeTensorFromDataId(e,t,n,r){n||=`float32`;let i={dataId:e,shape:t,dtype:n};return this.makeTensorFromTensorInfo(i,r)}makeTensorFromTensorInfo(e,t){let{dataId:n,shape:r,dtype:i}=e,a=new Ni(r,i,n,this.nextTensorId());return this.trackTensor(a,t),a}makeVariable(e,t=!0,n,r){n||=this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));let i=new Pi(e,t,n,this.nextTensorId());if(this.state.registeredVariables[i.name]!=null)throw Error(`Variable with name ${i.name} was already registered`);return this.state.registeredVariables[i.name]=i,this.incRef(i,this.backend),i}trackTensor(e,t){this.state.numTensors++,e.dtype===`string`&&this.state.numStringTensors++;let n=0;e.dtype!==`complex64`&&e.dtype!==`string`&&(n=e.size*ce(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof Pi||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;let t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype===`string`&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!==`complex64`&&e.dtype!==`string`){let t=e.size*ce(e.dtype);this.state.numBytes-=t}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(let e in this.state.registeredVariables){let t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){let e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons??=[],e.reasons.push(`Memory usage by string tensors is approximate (2 bytes per character)`)),e}async profile(e){this.state.profiling=!0;let t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(e=>e.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(let e of this.state.activeProfile.kernels)e.kernelTimeMs=await e.kernelTimeMs,e.extraInfo=await e.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,n,r,i,a){let o={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:i},s=Ir(e);s!=null&&(r=s.gradFunc),r!=null&&(o.gradient=e=>(e=e.map((e,t)=>{if(e==null){let e=n[t],r=be(e.size,e.dtype);return this.makeTensor(r,e.shape,e.dtype)}return e}),r(e.length>1?e:e[0],i,a))),this.state.activeTape.push(o)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){let t={track:[],name:`unnamed scope`,id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){let t=qi(e),n=new Set(t.map(e=>e.id));for(let e=0;e<this.state.activeScope.track.length;e++){let t=this.state.activeScope.track[e];!t.kept&&!n.has(t.id)&&t.dispose()}let r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(e=>{!e.kept&&e.scopeId===r.id&&this.track(e)})}gradients(e,t,n,r=!1){if(b(t.length>0,()=>`gradients() received an empty list of xs.`),n!=null&&n.dtype!==`float32`)throw Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);let i=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy(`forward`,e));b(i instanceof Ni,()=>`The result y returned by f() must be a tensor.`);let a=_i(this.state.activeTape,t,i);if(!r&&a.length===0&&t.length>0)throw Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.`);return this.tidy(`backward`,()=>{let e={};e[i.id]=n??$i(i.shape),vi(e,a,e=>this.tidy(e),ta);let r=t.map(t=>e[t.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(e=>{for(let t of e.saved)t.dispose()}),this.state.activeTape=null),{value:i,grads:r}})}customGrad(e){return b(me(e),()=>`The f passed in customGrad(f) must be a function.`),(...t)=>{b(t.every(e=>e instanceof Ni),()=>`The args passed in customGrad(f)(x1, x2,...) must all be tensors`);let n,r={};return t.forEach((e,t)=>{r[t]=e}),this.runKernelFunc({forwardFunc:(r,i)=>(n=e(...t,i),b(n.value instanceof Ni,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),b(me(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),backwardsFunc:(e,r)=>{let i=n.gradFunc(e,r),a=Array.isArray(i)?i:[i];b(a.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),b(a.every(e=>e instanceof Ni),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");let o={};return a.forEach((e,t)=>{o[t]=()=>e}),o},inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){let t=li(),n=await this.backend.time(e);return n.wallMs=li()-t,n}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Zi;for(let e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}};Qi.nextTensorId=0,Qi.nextVariableId=0;function $i(e){let t=ye(C(e),`float32`);return M.makeTensor(t,e,`float32`)}function ea(){let e=Pe();return e._tfengine??=new Qi(new De(e)),Me(e._tfengine.ENV),ji(()=>e._tfengine),e._tfengine}var M=ea();function ta(e,t){let n={a:e,b:t};return M.runKernel(`Add`,n)}function na(){return typeof navigator<`u`&&navigator!=null}function ra(e){if(e||na()){if(e||=navigator,e.product===`ReactNative`)return!0;let t=e.userAgent||e.vendor||(typeof window<`u`?window.opera:``);if(!t){let t=e;return t.userAgentData&&t.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function ia(){return typeof window<`u`&&window.document!=null||typeof WorkerGlobalScope<`u`}var aa=A();aa.registerFlag(`DEBUG`,()=>!1,e=>{e&&console.warn(`Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.`)}),aa.registerFlag(`IS_BROWSER`,()=>ia()),aa.registerFlag(`IS_NODE`,()=>typeof process<`u`&&process.versions!==void 0&&process.versions.node!==void 0),aa.registerFlag(`IS_CHROME`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),aa.registerFlag(`IS_SAFARI`,()=>typeof navigator<`u`&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor)),aa.registerFlag(`PROD`,()=>!1),aa.registerFlag(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`,()=>aa.getBool(`DEBUG`)),aa.registerFlag(`DEPRECATION_WARNINGS_ENABLED`,()=>!0),aa.registerFlag(`IS_TEST`,()=>!1),aa.registerFlag(`CHECK_COMPUTATION_FOR_ERRORS`,()=>aa.getBool(`DEBUG`)),aa.registerFlag(`WRAP_TO_IMAGEBITMAP`,()=>!1),aa.registerFlag(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`,()=>!1),aa.registerFlag(`USE_SETTIMEOUTCUSTOM`,()=>!1);function oa(e,t){let n=e;if(fi(e))return t===`string`?[]:[e.length];if(Ui(e)){let t=e.channels||`RGBA`;return[e.height,e.width*t.length]}if(Wi(e))return[e.buffer.size/(t==null?4:ce(t))];if(!Array.isArray(e))return[];let r=[];for(;Array.isArray(n)||fi(n)&&t!==`string`;)r.push(n.length),n=n[0];return Array.isArray(e)&&A().getBool(`TENSORLIKE_CHECK_SHAPE_CONSISTENCY`)&&sa(e,r,[]),r}function sa(e,t,n){if(n||=[],!Array.isArray(e)&&!fi(e)){b(t.length===0,()=>`Element arr[${n.join(`][`)}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}b(t.length>0,()=>`Element arr[${n.join(`][`)}] should be a primitive, but is an array of ${e.length} elements`),b(e.length===t[0],()=>`Element arr[${n.join(`][`)}] should have ${t[0]} elements, but has ${e.length} elements`);let r=t.slice(1);for(let t=0;t<e.length;++t)sa(e[t],r,n.concat(t))}function ca(e,t,n,r){if(e!==`string_or_numeric`){if(e==null)throw Error(`Expected dtype cannot be null.`);if(e!==`numeric`&&e!==t||e===`numeric`&&t===`string`)throw Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function N(e,t,n,r=`numeric`){if(e instanceof j())return ca(r,e.dtype,t,n),e;let i=pe(e);if(i!==`string`&&[`bool`,`int32`,`float32`].indexOf(r)>=0&&(i=r),ca(r,i,t,n),e==null||!fi(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`){let r=e==null?`null`:e.constructor.name;throw Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${r}'`)}let a=oa(e,i);!fi(e)&&!Array.isArray(e)&&(e=[e]);let o=i===`string`?pi(e,[],!0):ci(e,i);return M.makeTensor(o,a,i)}function la(e,t,n,r=`numeric`){if(!Array.isArray(e))throw Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((e,i)=>N(e,`${t}[${i}]`,n,r))}var ua=`__op`;function P(e){let t=Object.keys(e);if(t.length!==1)throw Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0],r=e[n];n.endsWith(`_`)&&(n=n.substring(0,n.length-1)),n+=ua;let i=(...e)=>{M.startScope(n);try{let t=r(...e);return Te(t)&&console.error(`Cannot return a Promise inside of tidy.`),M.endScope(t),t}catch(e){throw M.endScope(null),e}};return Object.defineProperty(i,"name",{value:n,configurable:!0}),i}function da(e,t){let n=N(e,`real`,`complex`),r=N(t,`imag`,`complex`);x(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);let i={real:n,imag:r};return M.runKernel(ot,i)}var fa=P({complex_:da});function pa(e,t,n,r){if(r==null)r=pe(e);else if(r===`complex64`)throw Error(`Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).`);if(Wi(e)||Ui(e)){if(r!==`float32`&&r!==`int32`)throw Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return M.backend.createTensorFromGPUData(e,t||n,r)}if(!fi(e)&&!Array.isArray(e)&&typeof e!=`number`&&typeof e!=`boolean`&&typeof e!=`string`)throw Error(`values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray`);if(t!=null){Se(t);let e=C(t),r=C(n);b(e===r,()=>`Based on the provided shape, [${t}], the tensor should have ${e} values but has ${r}`);for(let e=0;e<n.length;++e){let r=n[e],i=e!==n.length-1||r!==C(t.slice(e));b(n[e]===t[e]||!i,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!fi(e)&&!Array.isArray(e)&&(e=[e]),t||=n,e=r===`string`?pi(e,[],!0):ci(e,r),M.makeTensor(e,t,r)}function ma(e,t,n){return pa(e,t,oa(e,n),n)}var ha=class e{static join(t){return new e(t).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(e=>fi(e)?e.buffer:e),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let n=0;n<e.length;n++){let r=e[n];n!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);let i=t+r.byteLength;this.shards.push({buffer:r,start:t,end:i}),t=i}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(this.shards.length===0||(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),t=Math.min(this.byteLength,t),t<=e))return new ArrayBuffer(0);let n=this.findShardForByte(e);if(n===-1)throw Error(`Could not find start shard for byte ${e}`);let r=t-e,i=new ArrayBuffer(r),a=new Uint8Array(i),o=0;for(let r=n;r<this.shards.length;r++){let n=this.shards[r],i=e+o-n.start,s=o,c=Math.min(t,n.end)-n.start,l=new Uint8Array(n.buffer,i,c-i);if(a.set(l,s),o+=l.length,t<n.end)break}return i}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(t){return e<t.start?-1:+(e>=t.end)}if(t(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;let n=ga(this.shards,t);return n===-1?-1:(this.previousShardIndex=n,this.previousShardIndex)}};function ga(e,t){let n=0,r=e.length;for(;n<=r;){let i=Math.floor((r-n)/2)+n,a=t(e[i]);if(a===0)return i;a<0?r=i:n=i+1}return-1}function _a(){return M}function va(){return M.memory()}function F(e,t){return M.tidy(e,t)}function ya(e){qi(e).forEach(e=>e.dispose())}function ba(e){return M.keep(e)}function xa(e){return M.setBackend(e)}function Sa(){return M.ready()}function Ca(){return M.backendName}function wa(e,t,n=1){return M.registerBackend(e,t,n)}function Ta(){return M.backend}var Ea=4;async function Da(e,t){let n=[],r=[],i=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);for(let a=0;a<i.length;++a){let o=i[a],s=Array.isArray(e)?e[a].tensor:e[o];if(s.dtype!==`float32`&&s.dtype!==`int32`&&s.dtype!==`bool`&&s.dtype!==`string`&&s.dtype!==`complex64`)throw Error(`Unsupported dtype in weight '${o}': ${s.dtype}`);let c={name:o,shape:s.shape,dtype:s.dtype};if(s.dtype===`string`){let e=new Promise(async e=>{let t=await s.bytes(),n=t.reduce((e,t)=>e+t.length,0)+Ea*t.length,r=new Uint8Array(n),i=0;for(let e=0;e<t.length;e++){let n=t[e],a=new Uint8Array(new Uint32Array([n.length]).buffer);r.set(a,i),i+=Ea,r.set(n,i),i+=n.length}e(r)});r.push(e)}else r.push(s.data());t!=null&&(c.group=t),n.push(c)}return{data:Oa(await Promise.all(r)),specs:n}}function Oa(e){if(e===null)throw Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0,n=[];e.forEach(e=>{if(t+=e.byteLength,n.push(e.byteLength===e.buffer.byteLength?e:new e.constructor(e)),!(e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array))throw Error(`Unsupported TypedArray subtype: ${e.constructor.name}`)});let r=new Uint8Array(t),i=0;return n.forEach(e=>{r.set(new Uint8Array(e.buffer),i),i+=e.byteLength}),r.buffer}var ka=typeof Buffer<`u`&&(typeof Blob>`u`||typeof atob>`u`||typeof btoa>`u`);function Aa(e){return ka?Buffer.byteLength(e,`utf8`):new Blob([e]).size}function ja(e){if(ka)return Buffer.from(e).toString(`base64`);let t=new Uint8Array(e),n=``;for(let e=0,r=t.length;e<r;e++)n+=String.fromCharCode(t[e]);return btoa(n)}function Ma(e){if(ka){let t=Buffer.from(e,`base64`);return t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)}let t=atob(e),n=new Uint8Array(t.length);for(let e=0;e<t.length;++e)n.set([t.charCodeAt(e)],e);return n.buffer}function Na(e){return ha.join(e)}function Pa(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`Expected JSON model topology, received ArrayBuffer.`);return{dateSaved:new Date,modelTopologyType:`JSON`,modelTopologyBytes:e.modelTopology==null?0:Aa(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:Aa(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new ha(e.weightData).byteLength}}var Fa=class e{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return e.instance??=new e,e.instance}static registerSaveRouter(t){e.getInstance().saveRouters.push(t)}static registerLoadRouter(t){e.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return e.getHandlers(t,`save`)}static getLoadHandlers(t,n){return e.getHandlers(t,`load`,n)}static getHandlers(t,n,r){let i=[];return(n===`load`?e.getInstance().loadRouters:e.getInstance().saveRouters).forEach(e=>{let n=e(t,r);n!==null&&i.push(n)}),i}},Ia=e=>Fa.getSaveHandlers(e),La=`tensorflowjs`,Ra=1,za=`models_store`,Ba=`model_info_store`;function Va(){if(!A().getBool(`IS_BROWSER`))throw Error(`Failed to obtain IndexedDB factory because the current environmentis not a web browser.`);let e=typeof window>`u`?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw Error(`The current browser does not appear to support IndexedDB.`);return t}function Ha(e){let t=e.result;t.createObjectStore(za,{keyPath:`modelPath`}),t.createObjectStore(Ba,{keyPath:`modelPath`})}var Ua=class{constructor(e){if(this.indexedDB=Va(),e==null||!e)throw Error(`For IndexedDB, modelPath must not be null, undefined or empty.`);this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((e,n)=>{let r=this.indexedDB.open(La,Ra);r.onupgradeneeded=()=>Ha(r),r.onsuccess=()=>{let i=r.result;if(t==null){let t=i.transaction(za,`readonly`),r=t.objectStore(za).get(this.modelPath);r.onsuccess=()=>{if(r.result==null)return i.close(),n(Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));e(r.result.modelArtifacts)},r.onerror=e=>(i.close(),n(r.error)),t.oncomplete=()=>i.close()}else{t.weightData=ha.join(t.weightData);let r=Pa(t),a=i.transaction(Ba,`readwrite`),o=a.objectStore(Ba),s;try{s=o.put({modelPath:this.modelPath,modelArtifactsInfo:r})}catch(e){return n(e)}let c;s.onsuccess=()=>{c=i.transaction(za,`readwrite`);let s=c.objectStore(za),l;try{l=s.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:r})}catch(e){return n(e)}l.onsuccess=()=>e({modelArtifactsInfo:r}),l.onerror=e=>{o=a.objectStore(Ba);let t=o.delete(this.modelPath);t.onsuccess=()=>(i.close(),n(l.error)),t.onerror=e=>(i.close(),n(l.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}}},r.onerror=e=>n(r.error)})}};Ua.URL_SCHEME=`indexeddb://`;var Wa=e=>A().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(Ua.URL_SCHEME)?Ga(e.slice(Ua.URL_SCHEME.length)):null;Fa.registerSaveRouter(Wa),Fa.registerLoadRouter(Wa);function Ga(e){return new Ua(e)}function Ka(e){return e.startsWith(Ua.URL_SCHEME)?e.slice(Ua.URL_SCHEME.length):e}var qa=class{constructor(){this.indexedDB=Va()}async listModels(){return new Promise((e,t)=>{let n=this.indexedDB.open(La,Ra);n.onupgradeneeded=()=>Ha(n),n.onsuccess=()=>{let r=n.result,i=r.transaction(Ba,`readonly`),a=i.objectStore(Ba).getAll();a.onsuccess=()=>{let t={};for(let e of a.result)t[e.modelPath]=e.modelArtifactsInfo;e(t)},a.onerror=e=>(r.close(),t(a.error)),i.oncomplete=()=>r.close()},n.onerror=e=>t(n.error)})}async removeModel(e){return e=Ka(e),new Promise((t,n)=>{let r=this.indexedDB.open(La,Ra);r.onupgradeneeded=()=>Ha(r),r.onsuccess=()=>{let i=r.result,a=i.transaction(Ba,`readwrite`),o=a.objectStore(Ba),s=o.get(e),c;s.onsuccess=()=>{if(s.result==null)return i.close(),n(Error(`Cannot find model with path '${e}' in IndexedDB.`));{let r=o.delete(e),a=()=>{c=i.transaction(za,`readwrite`);let r=c.objectStore(za).delete(e);r.onsuccess=()=>t(s.result.modelArtifactsInfo),r.onerror=e=>n(s.error)};r.onsuccess=a,r.onerror=e=>(a(),i.close(),n(s.error))}},s.onerror=e=>(i.close(),n(s.error)),a.oncomplete=()=>{c==null?i.close():c.oncomplete=()=>i.close()}},r.onerror=e=>n(r.error)})}},Ja=`/`,Ya=`tensorflowjs_models`,Xa=`info`,Za=`model_topology`,Qa=`weight_specs`,$a=`weight_data`,eo=`model_metadata`;function to(e){return{info:[Ya,e,Xa].join(Ja),topology:[Ya,e,Za].join(Ja),weightSpecs:[Ya,e,Qa].join(Ja),weightData:[Ya,e,$a].join(Ja),modelMetadata:[Ya,e,eo].join(Ja)}}function no(e){for(let t of Object.values(e))window.localStorage.removeItem(t)}function ro(e){let t=e.split(Ja);if(t.length<3)throw Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(Ja)}function io(e){return e.startsWith(ao.URL_SCHEME)?e.slice(ao.URL_SCHEME.length):e}var ao=class{constructor(e){if(!A().getBool(`IS_BROWSER`)||typeof window>`u`||window.localStorage===void 0)throw Error(`The current environment does not support local storage.`);if(this.LS=window.localStorage,e==null||!e)throw Error(`For local storage, modelPath must not be null, undefined or empty.`);this.modelPath=e,this.keys=to(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw Error(`BrowserLocalStorage.save() does not support saving model topology in binary formats yet.`);{let t=JSON.stringify(e.modelTopology),n=JSON.stringify(e.weightSpecs),r=Pa(e),i=ha.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,ja(i));let a={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature==null?void 0:e.signature,userDefinedMetadata:e.userDefinedMetadata==null?void 0:e.userDefinedMetadata,modelInitializer:e.modelInitializer==null?void 0:e.modelInitializer,initializerSignature:e.initializerSignature==null?void 0:e.initializerSignature,trainingConfig:e.trainingConfig==null?void 0:e.trainingConfig};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:r}}catch{throw no(this.keys),Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){let e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!==`JSON`)throw Error(`BrowserLocalStorage does not support loading non-JSON model topology yet.`);let t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(n==null)throw Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;let r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;let i=this.LS.getItem(this.keys.modelMetadata);if(i!=null){let e=JSON.parse(i);t.format=e.format,t.generatedBy=e.generatedBy,t.convertedBy=e.convertedBy,e.signature!=null&&(t.signature=e.signature),e.userDefinedMetadata!=null&&(t.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(t.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(t.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(t.trainingConfig=e.trainingConfig)}let a=this.LS.getItem(this.keys.weightData);if(a==null)throw Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=Ma(a),t}};ao.URL_SCHEME=`localstorage://`;var oo=e=>A().getBool(`IS_BROWSER`)&&!Array.isArray(e)&&e.startsWith(ao.URL_SCHEME)?so(e.slice(ao.URL_SCHEME.length)):null;Fa.registerSaveRouter(oo),Fa.registerLoadRouter(oo);function so(e){return new ao(e)}var co=class{constructor(){b(A().getBool(`IS_BROWSER`),()=>`Current environment is not a web browser`),b(typeof window>`u`||window.localStorage!==void 0,()=>`Current browser does not appear to support localStorage`),this.LS=window.localStorage}async listModels(){let e={};for(let t=0;t<this.LS.length;++t){let n=this.LS.key(t);if(n.startsWith(`tensorflowjs_models/`)&&n.endsWith(`/info`)){let t=ro(n);e[t]=JSON.parse(this.LS.getItem(n))}}return e}async removeModel(e){e=io(e);let t=to(e);if(this.LS.getItem(t.info)==null)throw Error(`Cannot find model at path '${e}'`);let n=JSON.parse(this.LS.getItem(t.info));return no(t),n}},lo=`://`,uo=class e{constructor(){this.managers={}}static getInstance(){return e.instance??=new e,e.instance}static registerManager(t,n){b(t!=null,()=>`scheme must not be undefined or null.`),t.endsWith(lo)&&(t=t.slice(0,t.indexOf(lo))),b(t.length>0,()=>`scheme must not be an empty string.`);let r=e.getInstance();b(r.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),r.managers[t]=n}static getManager(t){let n=e.getInstance().managers[t];if(n==null)throw Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(e.getInstance().managers)}},fo=class{constructor(){this.messageName=`setTimeoutCustom`,this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder??=new TextEncoder,this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if(typeof window>`u`||!A().getBool(`USE_SETTIMEOUTCUSTOM`)){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},`*`)},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener(`message`,e=>{if(e.source===window&&e.data.name===this.messageName){e.stopPropagation();let t=this.functionRefs[e.data.index];t(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return Vr(e)}};if(A().get(`IS_BROWSER`)){A().setPlatform(`browser`,new fo);try{uo.registerManager(ao.URL_SCHEME,new co)}catch{}try{uo.registerManager(Ua.URL_SCHEME,new qa)}catch{}}var po=o(((e,t)=>{t.exports={}})),mo={importFetch:()=>po()},ho,go=class{constructor(){this.util=po(),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return A().global.fetch==null?(ho??=mo.importFetch(),ho(e,t)):A().global.fetch(e,t)}now(){let e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!==`utf-8`&&t!==`utf8`)throw Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?``:new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}};A().get(`IS_NODE`)&&!A().get(`IS_BROWSER`)&&A().setPlatform(`node`,new go);function _o(e,t=`float32`,n){return t||=`float32`,Se(e),new Oi(e,t,n)}function vo(e,t){let n=N(e,`x`,`cast`);if(!oe(t))throw Error(`Failed to cast to unknown dtype ${t}`);if(t===`string`&&n.dtype!==`string`||t!==`string`&&n.dtype===`string`)throw Error(`Only strings can be casted to strings`);let r={x:n},i={dtype:t};return M.runKernel(rt,r,i)}var I=P({cast_:vo});function yo(e){let t={x:N(e,`x`,`clone`,`string_or_numeric`)};return M.runKernel(Wt,t)}var bo=P({clone_:yo});function xo(e,t=!1){console.log(e.toString(t))}ea(),Mi({buffer:_o,cast:I,clone:bo,print:xo});function So(e,t){let n=N(e,`a`,`add`),r=N(t,`b`,`add`);[n,r]=Gi(n,r);let i={a:n,b:r};return M.runKernel(`Add`,i)}var L=P({add_:So});function Co(e,t){let n=N(e,`a`,`floorDiv`),r=N(t,`b`,`floorDiv`);[n,r]=Gi(n,r);let i={a:n,b:r};return M.runKernel(Rt,i)}var wo=P({floorDiv_:Co});function To(e,t){let n=N(e,`a`,`div`),r=N(t,`b`,`div`);if([n,r]=Gi(n,r),n.dtype===`int32`&&r.dtype===`int32`)return wo(n,r);let i={a:n,b:r};return M.runKernel(kt,i,{})}var R=P({div_:To});function Eo(e,t){let n=N(e,`a`,`mul`),r=N(t,`b`,`mul`);[n,r]=Gi(n,r);let i={a:n,b:r};return M.runKernel(_n,i)}var z=P({mul_:Eo});function Do(e){let t=N(e,`x`,`abs`);if(t.dtype===`complex64`){let e={x:t};return M.runKernel(st,e)}{let e={x:t};return M.runKernel(`Abs`,e)}}var Oo=P({abs_:Do});function ko(e){let t={x:N(e,`x`,`acos`)};return M.runKernel(Le,t)}var Ao=P({acos_:ko});function jo(e){let t={x:N(e,`x`,`acosh`)};return M.runKernel(Re,t)}var Mo=P({acosh_:jo});function No(e){b(Array.isArray(e),()=>`The argument passed to tf.addN() must be a list of tensors`),b(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);let t=e.map((e,t)=>N(e,`tensors${t}`,`addN`)),n=t[0];t.forEach(e=>{if(e.dtype!==n.dtype)throw Error(`All tensors passed to tf.addN() must have the same dtype`)}),t.forEach(e=>{if(!w(e.shape,n.shape))throw Error(`All tensors passed to tf.addN() must have the same shape`)});let r=t;return M.runKernel(ze,r)}var Po=P({addN_:No});function Fo(e,t=null,n=!1){let r={x:N(e,`x`,`all`,`bool`)},i={axis:t,keepDims:n};return M.runKernel(`All`,r,i)}var Io=P({all_:Fo});function Lo(e,t=null,n=!1){let r={x:N(e,`x`,`any`,`bool`)},i={axis:t,keepDims:n};return M.runKernel(`Any`,r,i)}var Ro=P({any_:Lo});function zo(e,t=0){let n={x:N(e,`x`,`argMax`)},r={axis:t};return M.runKernel(Be,n,r)}var Bo=P({argMax_:zo});function Vo(e,t=0){let n={x:N(e,`x`,`argMin`)},r={axis:t};return M.runKernel(Ve,n,r)}var Ho=P({argMin_:Vo});function Uo(e){let t={x:N(e,`x`,`asin`)};return M.runKernel(He,t)}var Wo=P({asin_:Uo});function Go(e){let t={x:N(e,`x`,`asinh`)};return M.runKernel(Ue,t)}var Ko=P({asinh_:Go});function qo(e){let t={x:N(e,`x`,`atan`)};return M.runKernel(We,t)}var Jo=P({atan_:qo});function Yo(e,t){let n=N(e,`a`,`atan2`),r=N(t,`b`,`atan2`);[n,r]=Gi(n,r);let i={a:n,b:r};return M.runKernel(Ke,i)}var Xo=P({atan2_:Yo});function Zo(e){let t={x:N(e,`x`,`atanh`)};return M.runKernel(Ge,t)}var Qo=P({atanh_:Zo});function $o(e,t,n,r,i=`NHWC`,a){let o=e[3];return ns(e,[...t,o],n,a,r,null,null,gs(i))}function es(e,t,n,r,i,a,o=`channelsLast`){let[s,c]=ss(t),l;if(o===`channelsLast`)l=[s,c,e[3],e[3]];else if(o===`channelsFirst`)l=[s,c,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return ns(e,l,n,r,i,a,!1,o)}function ts(e,t,n,r,i,a,o=`NDHWC`){let[s,c,l]=cs(t),u,d;if(o===`NDHWC`)d=`channelsLast`,u=[s,c,l,e[4],e[4]];else if(o===`NCDHW`)d=`channelsFirst`,u=[s,c,l,e[1],e[1]];else throw Error(`Unknown dataFormat ${o}`);return rs(e,u,n,r,i,!1,d,a)}function ns(e,t,n,r,i,a,o=!1,s=`channelsLast`){let[c,l,u,d]=[-1,-1,-1,-1];if(s===`channelsLast`)[c,l,u,d]=e;else if(s===`channelsFirst`)[c,d,l,u]=e;else throw Error(`Unknown dataFormat ${s}`);let[f,p,,m]=t,[h,g]=ss(n),[_,v]=ss(r),y=ls(f,_),b=ls(p,v),{padInfo:x,outHeight:S,outWidth:C}=us(i,l,u,h,g,y,b,a,s),w=o?m*d:m,T;return s===`channelsFirst`?T=[c,w,S,C]:s===`channelsLast`&&(T=[c,S,C,w]),{batchSize:c,dataFormat:s,inHeight:l,inWidth:u,inChannels:d,outHeight:S,outWidth:C,outChannels:w,padInfo:x,strideHeight:h,strideWidth:g,filterHeight:f,filterWidth:p,effectiveFilterHeight:y,effectiveFilterWidth:b,dilationHeight:_,dilationWidth:v,inShape:e,outShape:T,filterShape:t}}function rs(e,t,n,r,i,a=!1,o=`channelsLast`,s){let[c,l,u,d,f]=[-1,-1,-1,-1,-1];if(o===`channelsLast`)[c,l,u,d,f]=e;else if(o===`channelsFirst`)[c,f,l,u,d]=e;else throw Error(`Unknown dataFormat ${o}`);let[p,m,h,,g]=t,[_,v,y]=cs(n),[b,x,S]=cs(r),C=ls(p,b),w=ls(m,x),T=ls(h,S),{padInfo:E,outDepth:D,outHeight:ee,outWidth:te}=ds(i,l,u,d,_,v,y,C,w,T,s),O=a?g*f:g,ne;return o===`channelsFirst`?ne=[c,O,D,ee,te]:o===`channelsLast`&&(ne=[c,D,ee,te,O]),{batchSize:c,dataFormat:o,inDepth:l,inHeight:u,inWidth:d,inChannels:f,outDepth:D,outHeight:ee,outWidth:te,outChannels:O,padInfo:E,strideDepth:_,strideHeight:v,strideWidth:y,filterDepth:p,filterHeight:m,filterWidth:h,effectiveFilterDepth:C,effectiveFilterHeight:w,effectiveFilterWidth:T,dilationDepth:b,dilationHeight:x,dilationWidth:S,inShape:e,outShape:ne,filterShape:t}}function is(e,t,n,r,i){r??=os(e,t,n);let a=e[0],o=e[1];return[fs((a-t+2*r)/n+1,i),fs((o-t+2*r)/n+1,i)]}function as(e,t,n,r,i,a){i??=os(e,t[0],r[0]);let o=[0,0,0,n];for(let n=0;n<3;n++)e[n]+2*i>=t[n]&&(o[n]=fs((e[n]-t[n]+2*i)/r[n]+1,a));return o}function os(e,t,n,r=1){let i=ls(t,r);return Math.floor((e[0]*(n-1)-n+i)/2)}function ss(e){return typeof e==`number`?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function cs(e){return typeof e==`number`?[e,e,e]:e}function ls(e,t){return t<=1?e:e+(e-1)*(t-1)}function us(e,t,n,r,i,a,o,s,c){let l,u,d;if(typeof e==`number`){l={top:e,bottom:e,left:e,right:e,type:e===0?`VALID`:`NUMBER`};let i=is([t,n],a,r,e,s);u=i[0],d=i[1]}else if(e===`same`){u=Math.ceil(t/r),d=Math.ceil(n/i);let e=Math.max(0,(u-1)*r+a-t),s=Math.max(0,(d-1)*i+o-n),c=Math.floor(e/2),f=e-c,p=Math.floor(s/2);l={top:c,bottom:f,left:p,right:s-p,type:`SAME`}}else if(e===`valid`)l={top:0,bottom:0,left:0,right:0,type:`VALID`},u=Math.ceil((t-a+1)/r),d=Math.ceil((n-o+1)/i);else if(typeof e==`object`){let f=c===`channelsLast`?e[1][0]:e[2][0],p=c===`channelsLast`?e[1][1]:e[2][1],m=c===`channelsLast`?e[2][0]:e[3][0],h=c===`channelsLast`?e[2][1]:e[3][1];l={top:f,bottom:p,left:m,right:h,type:f===0&&p===0&&m===0&&h===0?`VALID`:`EXPLICIT`},u=fs((t-a+f+p)/r+1,s),d=fs((n-o+m+h)/i+1,s)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outHeight:u,outWidth:d}}function ds(e,t,n,r,i,a,o,s,c,l,u){let d,f,p,m;if(e===`valid`&&(e=0),typeof e==`number`){d={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?`VALID`:`NUMBER`};let h=as([t,n,r,1],[s,c,l],1,[i,a,o],e,u);f=h[0],p=h[1],m=h[2]}else if(e===`same`){f=Math.ceil(t/i),p=Math.ceil(n/a),m=Math.ceil(r/o);let e=(f-1)*i+s-t,u=(p-1)*a+c-n,h=(m-1)*o+l-r,g=Math.floor(e/2),_=e-g,v=Math.floor(u/2),y=u-v,b=Math.floor(h/2);d={top:v,bottom:y,left:b,right:h-b,front:g,back:_,type:`SAME`}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:d,outDepth:f,outHeight:p,outWidth:m}}function fs(e,t){if(!t)return Math.trunc(e);switch(t){case`round`:return Math.round(e);case`ceil`:return Math.ceil(e);case`floor`:return Math.floor(e);default:throw Error(`Unknown roundingMode ${t}`)}}function ps(e){let[t,n,r]=ss(e);return t===1&&n===1&&r===1}function ms(e,t){return ps(e)||ps(t)}function hs(e){return ss(e).every(e=>e>0)}function gs(e){if(e===`NHWC`)return`channelsLast`;if(e===`NCHW`)return`channelsFirst`;throw Error(`Unknown dataFormat ${e}`)}function _s(e,t,n){if(n!=null){if(typeof t==`string`)throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t==`number`)b(T(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t==`object`)t.forEach(t=>{t.forEach(t=>{b(T(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}function vs(e,t){let n={x:N(e,`x`,`reshape`,`string_or_numeric`)},r={shape:t};return M.runKernel(Fn,n,r)}var B=P({reshape_:vs});function ys(e,t,n,r,i){let a=N(e,`x`,`avgPool`,`float32`);b(ms(n,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`);let o=a,s=!1;a.rank===3&&(s=!0,o=B(a,[1,a.shape[0],a.shape[1],a.shape[2]])),b(o.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${o.rank}.`),_s(`avgPool`,r,i);let c={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:i},u=M.runKernel(qe,c,l);return u=I(u,a.dtype),s?B(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var bs=P({avgPool_:ys});function xs(e,t,n,r,i,a=`NDHWC`){let o=N(e,`x`,`avgPool3d`,`float32`),s=o,c=!1;o.rank===4&&(c=!0,s=B(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),b(s.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${s.rank}.`),b(a===`NDHWC`,()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),b(typeof n==`number`&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),_s(`avgPool3d`,r,i);let l={x:s},u={filterSize:t,strides:n,pad:r,dimRoundingMode:i,dataFormat:a},d=M.runKernel(Ye,l,u);return d=I(d,s.dtype),c?B(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}var Ss=P({avgPool3d_:xs});function Cs(e,t=0){b(e.length>=1,()=>`Pass at least one tensor to concat`);let n=la(e,`tensors`,`concat`,`string_or_numeric`);if(n[0].dtype===`complex64`&&n.forEach(e=>{if(e.dtype!==`complex64`)throw Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${e.dtype}. `)}),n.length===1)return bo(n[0]);let r=n,i={axis:t};return M.runKernel(ct,r,i)}var ws=P({concat_:Cs});function Ts(e,t,n=!1,r=!1){let i=N(e,`a`,`matMul`),a=N(t,`b`,`matMul`);[i,a]=Gi(i,a);let o={a:i,b:a},s={transposeA:n,transposeB:r};return M.runKernel(Ze,o,s)}var Es=P({matMul_:Ts});function Ds(e){let t={x:N(e,`x`,`sigmoid`,`float32`)};return M.runKernel(Qn,t)}var Os=P({sigmoid_:Ds});function ks(e,t,n){let r=N(e,`x`,`slice`,`string_or_numeric`);if(r.rank===0)throw Error(`Slicing scalar is not possible`);let i={x:r},a={begin:t,size:n};return M.runKernel(Yn,i,a)}var As=P({slice_:ks});function js(e){let t={x:N(e,`x`,`tanh`,`float32`)};return M.runKernel(gr,t)}var Ms=P({tanh_:js});function Ns(e,t,n){let r=N(e,`x`,`batchToSpaceND`),i=t.reduce((e,t)=>e*t);b(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),b(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),b(r.shape[0]%i===0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(` * `)} === ${i}`);let a={x:r},o={blockShape:t,crops:n};return M.runKernel(Qe,a,o)}var Ps=P({batchToSpaceND_:Ns});function Fs(e){let t;return t=e.rank===0||e.rank===1?B(e,[1,1,1,e.size]):e.rank===2?B(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?B(e,[1,e.shape[0],e.shape[1],e.shape[2]]):e,t}function Is(e,t,n,r,i,a){a??=.001;let o=N(e,`x`,`batchNorm`),s=N(t,`mean`,`batchNorm`),c=N(n,`variance`,`batchNorm`),l;i!=null&&(l=N(i,`scale`,`batchNorm`));let u;r!=null&&(u=N(r,`offset`,`batchNorm`)),b(s.rank===c.rank,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),b(u==null||s.rank===u.rank,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),b(l==null||s.rank===l.rank,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`);let d={x:Fs(o),scale:l,offset:u,mean:s,variance:c},f={varianceEpsilon:a};return B(M.runKernel(zt,d,f),o.shape)}var Ls=P({batchNorm_:Is});function Rs(e,t,n,r,i,a){let o=N(e,`x`,`batchNorm`),s=N(t,`mean`,`batchNorm`),c=N(n,`variance`,`batchNorm`),l;i!=null&&(l=N(i,`scale`,`batchNorm`));let u;return r!=null&&(u=N(r,`offset`,`batchNorm`)),b(o.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),b(s.rank===2||s.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${s.rank}.`),b(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${c.rank}.`),l!=null&&b(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${l.rank}.`),u!=null&&b(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${u.rank}.`),Ls(o,s,c,u,l,a)}var zs=P({batchNorm2d_:Rs});function Bs(e,t,n,r,i,a){let o=N(e,`x`,`batchNorm`),s=N(t,`mean`,`batchNorm`),c=N(n,`variance`,`batchNorm`),l;i!=null&&(l=N(i,`scale`,`batchNorm`));let u;return r!=null&&(u=N(r,`offset`,`batchNorm`)),b(o.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),b(s.rank===3||s.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${s.rank}.`),b(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${c.rank}.`),l!=null&&b(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${l.rank}.`),u!=null&&b(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${u.rank}.`),Ls(o,s,c,u,l,a)}var Vs=P({batchNorm3d_:Bs});function Hs(e,t,n,r,i,a){let o=N(e,`x`,`batchNorm`),s=N(t,`mean`,`batchNorm`),c=N(n,`variance`,`batchNorm`),l;i!=null&&(l=N(i,`scale`,`batchNorm`));let u;return r!=null&&(u=N(r,`offset`,`batchNorm`)),b(o.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),b(s.rank===4||s.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${s.rank}.`),b(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${c.rank}.`),l!=null&&b(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${l.rank}.`),u!=null&&b(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${u.rank}.`),Ls(o,s,c,u,l,a)}var Us=P({batchNorm4d_:Hs});function Ws(e,t,n){let r=N(e,`x`,`bincount`),i=N(t,`weights`,`bincount`);b(r.dtype===`int32`,()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),b(n>=0,()=>`size must be non-negative, but got ${n}.`),b(i.size===r.size||i.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${i.shape}.`);let a={x:r,weights:i},o={size:n};return M.runKernel($e,a,o)}var Gs=P({bincount_:Ws});function Ks(e,t){let n=N(e,`broadcastTo`,`x`),r=n.shape;if(Se(t),t.length<n.rank)throw Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){let e=n.shape.slice();for(;e.length<t.length;)e.unshift(1);n=B(n,e)}let i=n.shape,a=Array.from(t);for(let e=t.length-1;e>=0;e--)if(i[e]===t[e])a[e]=1;else if(n.shape[e]!==1)throw Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(a.map((e,t)=>e>1?t:-1).filter(e=>e>=0).length===0)return bo(n);let o={x:n},s={reps:a};return M.runKernel(_r,o,s)}var qs=P({broadcastTo_:Ks});function Js(e){let t={x:N(e,`x`,`ceil`,`float32`)};return M.runKernel(it,t)}var Ys=P({ceil_:Js});function Xs(e,t,n){Se(e),n||=pe(t);let r={shape:e,value:t,dtype:n};return M.runKernel(Ft,{},r)}function Zs(e,t,n){let r=N(e,`x`,`clipByValue`);if(b(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return Xs(r.shape,t,r.dtype);let i={x:r},a={clipValueMin:t,clipValueMax:n};return M.runKernel(at,i,a)}var Qs=P({clipByValue_:Zs});function $s(e){return ws(e,0)}var ec=P({concat1d_:$s});function tc(e,t){return ws(e,t)}var nc=P({concat2d_:tc});function rc(e,t){return ws(e,t)}var ic=P({concat3d_:rc});function ac(e,t){return ws(e,t)}var oc=P({concat4d_:ac});function sc(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=N(e,`x`,`conv2d`,`float32`),c=N(t,`filter`,`conv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=B(s,[1,s.shape[0],s.shape[1],s.shape[2]])),b(l.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),b(c.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),_s(`conv2d`,r,o);let d=i===`NHWC`?l.shape[3]:l.shape[1];b(d===c.shape[2],()=>`Error in conv2d: depth of input (${d}) must match input depth for filter ${c.shape[2]}.`),b(ms(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),b(hs(a),()=>`Error in conv2D: Dilated rates should be larger than 0.`),b(hs(n),()=>`Error in conv2D: Strides should be larger than 0.`);let f={x:l,filter:c},p={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},m=M.runKernel(lt,f,p);return u?B(m,[m.shape[1],m.shape[2],m.shape[3]]):m}var cc=P({conv2d_:sc});function lc(e,t,n,r,i=`NWC`,a=1,o){let s=N(e,`x`,`conv1d`),c=N(t,`filter`,`conv1d`),l=s,u=!1;s.rank===2&&(u=!0,l=B(s,[1,s.shape[0],s.shape[1]])),b(l.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${l.rank}.`),b(c.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${c.rank}.`),_s(`conv1d`,r,o),b(l.shape[2]===c.shape[1],()=>`Error in conv1d: depth of input (${l.shape[2]}) must match input depth for filter ${c.shape[1]}.`),b(ms(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),b(hs(a),()=>`Error in conv1D: Dilated rates should be larger than 0.`),b(hs(n),()=>`Error in conv1D: Stride should be larger than 0.`),b(i===`NWC`,()=>`Error in conv1d: got dataFormat of ${i} but only NWC is currently supported.`);let d=B(c,[1,c.shape[0],c.shape[1],c.shape[2]]),f=cc(B(l,[l.shape[0],1,l.shape[1],l.shape[2]]),d,[1,n],r,`NHWC`,[1,a],o);return u?B(f,[f.shape[2],f.shape[3]]):B(f,[f.shape[0],f.shape[2],f.shape[3]])}var uc=P({conv1d_:lc});function dc(e,t,n,r,i,a=`NHWC`,o){b(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let s=e,c=t,l=!1;t.rank===3&&(l=!0,c=B(t,[1,t.shape[0],t.shape[1],t.shape[2]]),s=[1,e[0],e[1],e[2]]),b(s.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${s.length}.`),b(c.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${c.rank}`),b(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);let u=a===`NHWC`?s[3]:s[1],d=a===`NHWC`?c.shape[3]:c.shape[1];b(u===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[2]}.`),b(d===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${d}) must match output depth for filter ${n.shape[3]}.`),_s(`conv2dDerInput`,i,o);let f={dy:c,filter:n},p={strides:r,pad:i,dataFormat:a,dimRoundingMode:o,inputShape:s},m=M.runKernel(dt,f,p);return l?B(m,[m.shape[1],m.shape[2],m.shape[3]]):m}var fc=P({conv2DBackpropInput_:dc});function pc(e,t,n,r,i,a){return fc(n,N(e,`x`,`conv2dTranspose`),N(t,`filter`,`conv2dTranspose`),r,i,`NHWC`,a)}var mc=P({conv2dTranspose_:pc});function hc(e,t,n,r,i=`NDHWC`,a=[1,1,1]){let o=N(e,`x`,`conv3d`),s=N(t,`filter`,`conv3d`),c=o,l=!1;o.rank===4&&(l=!0,c=B(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),b(c.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${c.rank}.`),b(s.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${s.rank}.`),b(c.shape[4]===s.shape[3],()=>`Error in conv3d: depth of input (${c.shape[4]}) must match input depth for filter ${s.shape[3]}.`),b(ms(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),b(i===`NDHWC`,()=>`Error in conv3d: got dataFormat of ${i} but only NDHWC is currently supported.`),b(hs(a),()=>`Error in conv3D: Dilated rates should be larger than 0.`),b(hs(n),()=>`Error in conv3D: Strides should be larger than 0.`);let u={x:c,filter:s},d={strides:n,pad:r,dataFormat:i,dilations:a},f=M.runKernel(ft,u,d);return l?B(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}var gc=P({conv3d_:hc});function _c(e,t,n,r,i){b(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,o=t,s=!1;t.rank===4&&(s=!0,o=B(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),a=[1,e[0],e[1],e[2],e[3]]);let c=a[4],l=o.shape[4];b(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),b(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),b(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),b(c===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${c}) must match input depth for filter ${n.shape[3]}.`),b(l===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[4]}.`);let u={dy:o,filter:n},d={pad:i,strides:r,inputShape:a},f=M.runKernel(mt,u,d);return s?B(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}var vc=P({conv3DBackpropInput_:_c});function yc(e,t,n,r,i){return vc(n,N(e,`x`,`conv3dTranspose`),N(t,`filter`,`conv3dTranspose`),r,i)}var bc=P({conv3dTranspose_:yc});function xc(e){let t={x:N(e,`x`,`cos`,`float32`)};return M.runKernel(`Cos`,t)}var Sc=P({cos_:xc});function Cc(e){let t={x:N(e,`x`,`cosh`,`float32`)};return M.runKernel(ht,t)}var wc=P({cosh_:Cc});function Tc(e,t=0,n=!1,r=!1){let i={x:N(e,`x`,`cumprod`)},a={axis:t,exclusive:n,reverse:r};return M.runKernel(gt,i,a)}var Ec=P({cumprod_:Tc});function Dc(e,t=0,n=!1,r=!1){let i={x:N(e,`x`,`cumsum`)},a={axis:t,exclusive:n,reverse:r};return M.runKernel(_t,i,a)}var Oc=P({cumsum_:Dc});function kc(e,t,n,r=!1){let i=N(e,`x`,`denseBincount`),a=N(t,`weights`,`denseBincount`);b(i.dtype===`int32`,()=>`Error in denseBincount: input dtype must be int32, but got ${i.dtype}`),b(i.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${i.rank}.`),b(n>=0,()=>`size must be non-negative, but got ${n}.`),b(a.size===i.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${i.shape}, weights shape: ${a.shape}.`);let o={x:i,weights:a},s={size:n,binaryOutput:r};return M.runKernel(yt,o,s)}var Ac=P({denseBincount_:kc});function jc(e,t,n=`NHWC`){let r=N(e,`x`,`depthToSpace`,`float32`),i=n===`NHWC`?r.shape[1]:r.shape[2],a=n===`NHWC`?r.shape[2]:r.shape[3],o=n===`NHWC`?r.shape[3]:r.shape[1];b(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),b(i*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${i} and ${t}  for depthToSpace with input shape
    ${r.shape}`),b(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${t} for depthToSpace with input shape
        ${r.shape}`),b(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${r.shape}`);let s={x:r},c={blockSize:t,dataFormat:n};return M.runKernel(bt,s,c)}var Mc=P({depthToSpace_:jc});function Nc(e,t,n,r,i=`NHWC`,a=[1,1],o){let s=N(e,`x`,`depthwiseConv2d`,`float32`),c=N(t,`filter`,`depthwiseConv2d`,`float32`),l=s,u=!1;s.rank===3&&(u=!0,l=B(s,[1,s.shape[0],s.shape[1],s.shape[2]])),b(l.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${l.rank}.`),b(c.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${c.rank}.`);let d=i===`NHWC`?l.shape[3]:l.shape[1];b(d===c.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${d}) must match the inChannels dimension in filter ${c.shape[2]}.`),_s(`depthwiseConv2d`,r,o);let f={x:l,filter:c},p={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o},m=M.runKernel(xt,f,p);return u?B(m,[m.shape[1],m.shape[2],m.shape[3]]):m}var Pc=P({depthwiseConv2d_:Nc});function Fc(e,t,n,r,i=[1,1],a=`NHWC`){let o=N(e,`x`,`dilation2d`),s=N(t,`filter`,`dilation2d`);b(o.rank===3||o.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),b(s.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${s.rank}.`),b(a===`NHWC`,()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let c=o,l=!1;o.rank===3&&(c=B(o,[1,o.shape[0],o.shape[1],o.shape[2]]),l=!0),b(c.shape[3]===s.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${c.shape[3]} vs ${s.shape[2]}`);let u={x:c,filter:s},d={strides:n,pad:r,dilations:i},f=M.runKernel(Tt,u,d);return l?B(f,[f.shape[1],f.shape[2],f.shape[3]]):f}var Ic=P({dilation2d_:Fc});function Lc(e,t){let n=e.length,r=[];for(let i=0;i<n;i++){let a=n-1-i,o=e[a]||1;(t[t.length-1-i]||1)>1&&o===1&&r.unshift(a)}return r}function Rc(e,t){let n=[];for(let r=0;r<t.length;r++){let i=e[e.length-r-1],a=t.length-r-1,o=t[a];(i==null||i===1&&o>1)&&n.unshift(a)}return n}function V(e,t){let n=Math.max(e.length,t.length),r=Array(n);for(let i=0;i<n;i++){let a=e[e.length-i-1];a??=1;let o=t[t.length-i-1];if(o??=1,a===1)r[n-i-1]=o;else if(o===1)r[n-i-1]=a;else if(a!==o){let n=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(n)}else r[n-i-1]=a}return r}function zc(e,t){let n=N(e,`a`,`equal`,`string_or_numeric`),r=N(t,`b`,`equal`,`string_or_numeric`);[n,r]=Gi(n,r),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(Mt,i)}var Bc=P({equal_:zc});function Vc(e,t,n){let r=N(t,`a`,`where`),i=N(n,`b`,`where`),a=N(e,`condition`,`where`,`bool`),o=V(V(a.shape,r.shape),i.shape),s={condition:qs(a,o),t:qs(r,o),e:qs(i,o)};return M.runKernel(qn,s)}var Hc=P({where_:Vc});function Uc(e){let t={x:N(e,`x`,`zerosLike`)};return M.runKernel(wr,t)}var Wc=P({zerosLike_:Uc});function Gc(e,t){let n=N(e,`a`,`div`),r=N(t,`b`,`div`);[n,r]=Gi(n,r);let i=R(n,r),a=Wc(i);return Hc(Bc(r,a),a,i)}var Kc=P({divNoNan_:Gc});function qc(e,t){let n=N(e,`t1`,`dot`),r=N(t,`t2`,`dot`);b((n.rank===1||n.rank===2)&&(r.rank===1||r.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);let i=n.rank===1?n.size:n.shape[1],a=r.rank===1?r.size:r.shape[0];if(b(i===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${i} and ${a}.`),n.rank===1&&r.rank===1)return B(Es(B(n,[1,-1]),B(r,[-1,1])),[]);if(n.rank===1&&r.rank===2){let e=Es(B(n,[1,-1]),B(r,[r.shape[0],r.shape[1]]));return B(e,[e.size])}if(n.rank===2&&r.rank===1){let e=Es(n,B(r,[-1,1]));return B(e,[e.size])}return Es(n,B(r,[r.shape[0],r.shape[1]]))}var Jc=P({dot_:qc});function Yc(e,...t){let n=t.map((e,t)=>N(e,`tensors${t}`,`einsum`)),r={equation:e};return M.runKernel(At,n,r)}var Xc=P({einsum_:Yc});function Zc(e){let t={x:N(e,`x`,`elu`,`float32`)};return M.runKernel(`Elu`,t)}var Qc=P({elu_:Zc});function $c(e){let t=N(e,`x`,`erf`);b(t.dtype===`int32`||t.dtype===`float32`,()=>"Input dtype must be `int32` or `float32`."),t.dtype===`int32`&&(t=I(t,`float32`));let n={x:t};return M.runKernel(`Erf`,n)}var el=P({erf_:$c});function tl(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function nl(e,t,n){let r=e.length+t.length,i=[],a=0,o=0;for(let s=0;s<r;s++)n.indexOf(s)===-1?i.push(e[a++]):i.push(t[o++]);return i}function rl(e,t){let n=[],r=e.length;for(let i=0;i<r;i++)t.indexOf(i)===-1&&n.push(e[i]);return[n,t.map(t=>e[t])]}function il(e,t){return nl(e,t.map(e=>1),t)}function al(e,t,n){b(tl(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function ol(e,t){if(tl(e,t))return null;let n=[];for(let r=0;r<t;++r)e.indexOf(r)===-1&&n.push(r);return e.forEach(e=>n.push(e)),n}function sl(e){return e.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function cl(e,t){let n=[];for(let r=t-e;r<t;++r)n.push(r);return n}function ll(e,t=null,n=!1){let r={x:N(e,`x`,`max`)},i={reductionIndices:t,keepDims:n};return M.runKernel(`Max`,r,i)}var ul=P({max_:ll});function dl(e,t=null,n=!1){let r={x:N(e,`x`,`min`)},i={axis:t,keepDims:n};return M.runKernel(`Min`,r,i)}var fl=P({min_:dl});function pl(e,t){let n=N(e,`base`,`pow`),r=N(t,`exp`,`pow`);[n,r]=Gi(n,r);let i={a:n,b:r};return M.runKernel(`Pow`,i)}var ml=P({pow_:pl});function H(e,t){if((fi(e)&&t!==`string`||Array.isArray(e))&&t!==`complex64`)throw Error(`Error creating a new Scalar: value must be a primitive (number|boolean|string)`);if(t===`string`&&fi(e)&&!(e instanceof Uint8Array))throw Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return pa(e,[],[],t)}function hl(e){let t={x:N(e,`x`,`sqrt`,`float32`)};return M.runKernel(er,t)}var gl=P({sqrt_:hl});function _l(e){let t=N(e,`x`,`square`);return M.runKernel(`Square`,{x:t},{})}var vl=P({square_:_l});function yl(e,t=null,n=!1){let r=N(e,`x`,`sum`);r.dtype===`bool`&&(r=I(r,`int32`));let i={x:r},a={axis:t,keepDims:n};return M.runKernel(`Sum`,i,a)}var U=P({sum_:yl});function bl(e,t=`euclidean`,n=null,r=!1){e=N(e,`x`,`norm`);let i=xl(e,t,n),a=i.shape;if(r){let t=O(n,e.shape);a=il(i.shape,t)}return B(i,a)}function xl(e,t,n=null){if(e.rank===0)return Oo(e);if(e.rank!==1&&n===null)return xl(B(e,[-1]),t,n);if(e.rank===1||typeof n==`number`||Array.isArray(n)&&n.length===1){if(t===1)return U(Oo(e),n);if(t===1/0)return ul(Oo(e),n);if(t===-1/0)return fl(Oo(e),n);if(t===`euclidean`||t===2)return gl(U(ml(Oo(e),H(2,`int32`)),n));throw Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return ul(U(Oo(e),n[0]),n[1]-1);if(t===1/0)return ul(U(Oo(e),n[1]),n[0]);if(t===-1/0)return fl(U(Oo(e),n[1]),n[0]);if(t===`fro`||t===`euclidean`)return gl(U(vl(e),n));throw Error(`Error in norm: invalid ord value: ${t}`)}throw Error(`Error in norm: invalid axis: ${n}`)}var Sl=P({norm_:bl});function Cl(e,t=null,n=!1){return Sl(e,`euclidean`,t,n)}var wl=P({euclideanNorm_:Cl});function Tl(e){let t={x:N(e,`x`,`exp`)};return M.runKernel(`Exp`,t)}var El=P({exp_:Tl});function Dl(e,t=0){let n=N(e,`x`,`expandDims`,`string_or_numeric`);b(t<=n.rank,()=>`Axis must be <= rank of the tensor`);let r={input:n},i={dim:t};return M.runKernel(Nt,r,i)}var Ol=P({expandDims_:Dl});function kl(e){let t={x:N(e,`x`,`expm1`)};return M.runKernel(Pt,t)}var Al=P({expm1_:kl});function jl(e,t){let n=N(e,`x`,`tile`,`string_or_numeric`);b(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);let r={x:n},i={reps:t};return M.runKernel(_r,r,i)}var Ml=P({tile_:jl});function Nl(e,t,n,r=`float32`){t??=e;let i=_o([e,t],r),a=e<=t?e:t;for(let e=0;e<a;++e)i.set(1,e,e);let o=B(i.toTensor(),[e,t]);if(n==null)return o;if(n.length===1)return Ml(Ol(o,0),[n[0],1,1]);if(n.length===2)return Ml(Ol(Ol(o,0),0),[n[0],n[1],1,1]);if(n.length===3)return Ml(Ol(Ol(Ol(o,0),0),0),[n[0],n[1],n[2],1,1]);throw Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}var Pl=P({eye_:Nl});function Fl(e){let t={x:N(e,`x`,`floor`,`float32`)};return M.runKernel(Lt,t)}var Il=P({floor_:Fl});function Ll(e,t,n=0,r=0){let i={x:N(e,`x`,`gather`),indices:N(t,`indices`,`gather`,`int32`)},a={axis:n,batchDims:r};return M.runKernel(Bt,i,a)}var Rl=P({gather_:Ll});function zl(e,t){let n=N(e,`a`,`greater`,`string_or_numeric`),r=N(t,`b`,`greater`,`string_or_numeric`);[n,r]=Gi(n,r),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(Ht,i)}var Bl=P({greater_:zl});function Vl(e,t){let n=N(e,`a`,`greaterEqual`,`string_or_numeric`),r=N(t,`b`,`greaterEqual`,`string_or_numeric`);[n,r]=Gi(n,r),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(Ut,i)}var Hl=P({greaterEqual_:Vl});function Ul(e){let t={input:N(e,`input`,`imag`)};return M.runKernel(Kt,t)}var Wl=P({imag_:Ul});function Gl(e){let t={x:N(e,`x`,`isFinite`)};return M.runKernel(qt,t)}var Kl=P({isFinite_:Gl});function ql(e){let t={x:N(e,`x`,`isInf`)};return M.runKernel(Jt,t)}var Jl=P({isInf_:ql});function Yl(e){let t={x:N(e,`x`,`isNaN`)};return M.runKernel(Yt,t)}var Xl=P({isNaN_:Yl});function Zl(e,t=.2){let n={x:N(e,`x`,`leakyRelu`)},r={alpha:t};return M.runKernel(Xt,n,r)}var Ql=P({leakyRelu_:Zl});function $l(e,t){let n=N(e,`a`,`less`,`string_or_numeric`),r=N(t,`b`,`less`,`string_or_numeric`);[n,r]=Gi(n,r),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(Zt,i)}var eu=P({less_:$l});function tu(e,t){let n=N(e,`a`,`lessEqual`,`string_or_numeric`),r=N(t,`b`,`lessEqual`,`string_or_numeric`);[n,r]=Gi(n,r),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(Qt,i)}var nu=P({lessEqual_:tu});function ru(e,t=5,n=1,r=1,i=.5){let a=N(e,`x`,`localResponseNormalization`);b(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),b(T(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=a,s=!1;a.rank===3&&(s=!0,o=B(a,[1,a.shape[0],a.shape[1],a.shape[2]]));let c={x:o},l={depthRadius:t,bias:n,alpha:r,beta:i},u=M.runKernel(`LRN`,c,l);return s?B(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var iu=P({localResponseNormalization_:ru});function au(e){let t={x:N(e,`x`,`log`,`float32`)};return M.runKernel(`Log`,t)}var ou=P({log_:au});function su(e){let t={x:N(e,`x`,`log1p`)};return M.runKernel(en,t)}var cu=P({log1p_:su});function lu(e,t){b(me(e),()=>`The f passed in variableGrads(f) must be a function`),b(t==null||Array.isArray(t)&&t.every(e=>e instanceof Pi),()=>`The varList passed in variableGrads(f, varList) must be an array of variables`);let n=t!=null;if(!n){t=[];for(let e in M.registeredVariables)t.push(M.registeredVariables[e])}let r=n?t.filter(e=>!e.trainable):null,i=t.length;t=t.filter(e=>e.trainable),b(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${i} variables is trainable.`);let{value:a,grads:o}=M.gradients(e,t,null,!0);b(o.some(e=>e!=null),()=>`Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize().`),b(a.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${a.rank} tensor`);let s={};return t.forEach((e,t)=>{o[t]!=null&&(s[e.name]=o[t])}),r?.forEach(e=>s[e.name]=null),{value:a,grads:s}}function uu(e){return M.customGrad(e)}function du(e){let t={x:N(e,`x`,`neg`)};return M.runKernel(`Neg`,t)}var fu=P({neg_:du});function pu(e){let t={x:N(e,`x`,`softplus`)};return M.runKernel($n,t)}var mu=P({softplus_:pu});function hu(e){let t=N(e,`x`,`logSigmoid`);return uu(e=>({value:fu(mu(fu(e))),gradFunc:t=>z(t,Os(fu(e)))}))(t)}var gu=P({logSigmoid_:hu});function _u(e,t){let n=N(e,`a`,`sub`),r=N(t,`b`,`sub`);[n,r]=Gi(n,r);let i={a:n,b:r};return M.runKernel(`Sub`,i)}var W=P({sub_:_u});function vu(e,t=-1){let n=N(e,`logits`,`logSoftmax`);if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return uu((e,n)=>{let r=W(e,ul(e,t,!0)),i=W(I(r,`float32`),ou(U(El(r),t,!0)));return n([i]),{value:i,gradFunc:(e,n)=>{let[r]=n,i=El(r);return W(e,z(U(e,t,!0),i))}}})(n)}var yu=P({logSoftmax_:vu});function bu(e,t=null,n=!1){let r=N(e,`x`,`logSumExp`),i=O(t,r.shape),a=ul(r,i,!0),o=ou(U(El(W(r,a)),i)),s=L(B(a,o.shape),o);return n?B(s,il(s.shape,i)):s}var xu=P({logSumExp_:bu});function Su(e,t){let n=N(e,`a`,`logicalAnd`,`bool`),r=N(t,`b`,`logicalAnd`,`bool`);V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(tn,i)}var Cu=P({logicalAnd_:Su});function wu(e){let t={x:N(e,`x`,`logicalNot`,`bool`)};return M.runKernel(nn,t)}var Tu=P({logicalNot_:wu});function Eu(e,t){let n=N(e,`a`,`logicalOr`,`bool`),r=N(t,`b`,`logicalOr`,`bool`);V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(rn,i)}var Du=P({logicalOr_:Eu});function Ou(e,t){let n=N(e,`a`,`logicalXor`,`bool`),r=N(t,`b`,`logicalXor`,`bool`);return V(n.shape,r.shape),Cu(Du(e,t),Tu(Cu(e,t)))}var ku=P({logicalXor_:Ou});function Au(e,t,n,r,i){let a=N(e,`x`,`maxPool`),o=a,s=!1;a.rank===3&&(s=!0,o=B(a,[1,a.shape[0],a.shape[1],a.shape[2]])),b(o.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${o.rank}.`),b(ms(n,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`),_s(`maxPool`,r,i);let c={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:i},u=M.runKernel(cn,c,l);return s?B(u,[u.shape[1],u.shape[2],u.shape[3]]):u}var ju=P({maxPool_:Au});function Mu(e,t=[1,1,1],n,r,i,a=`NDHWC`){let o=N(e,`x`,`maxPool3d`),s=o,c=!1;o.rank===4&&(c=!0,s=B(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),b(s.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${s.rank}.`),b(a===`NDHWC`,()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),_s(`maxPool3d`,r,i);let l={x:s},u={filterSize:t,strides:n,pad:r,dimRoundingMode:i,dataFormat:a},d=M.runKernel(un,l,u);return c?B(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}var Nu=P({maxPool3d_:Mu});function Pu(e,t){let n=N(e,`a`,`maximum`),r=N(t,`b`,`maximum`);[n,r]=Gi(n,r),n.dtype===`bool`&&(n=I(n,`int32`),r=I(r,`int32`)),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(sn,i)}var Fu=P({maximum_:Pu});function Iu(e,t=null,n=!1){let r={x:N(e,`x`,`mean`)},i={axis:t,keepDims:n};return M.runKernel(pn,r,i)}var Lu=P({mean_:Iu});function Ru(e,t=`float32`){if(Se(e),t===`complex64`)return fa(Ru(e,`float32`),Ru(e,`float32`));let n=be(C(e),t);return M.makeTensor(n,e,t)}function zu(e,t=`float32`){if(Se(e),t===`complex64`)return fa(zu(e,`float32`),Ru(e,`float32`));let n=ye(C(e),t);return M.makeTensor(n,e,t)}function Bu(e,t){let n=N(e,`a`,`minimum`),r=N(t,`b`,`minimum`);[n,r]=Gi(n,r),n.dtype===`bool`&&(n=I(n,`int32`),r=I(r,`int32`)),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(mn,i)}var Vu=P({minimum_:Bu});function Hu(e,t,n){b(n===`reflect`||n===`symmetric`,()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);let r=N(e,`x`,`mirrorPad`);if(r.rank===0)throw Error(`mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad`);b(t.length===r.rank,()=>`Padding doesn't match input. Must be ${r.rank}. Got ${t.length}.`);let i=+(n===`reflect`);for(let e=0;e<r.rank;e++)b(t[e].length===2,()=>`Invalid number of paddings. Must be length of 2 each.`),b(t[e][0]>=0&&t[e][0]<=r.shape[e]-i&&t[e][1]>=0&&t[e][1]<=r.shape[e]-i,()=>`Padding in dimension ${e} cannot be greater than or equal to ${r.shape[e]-i} or less than 0 for input of shape ${r.shape}`);let a={paddings:t,mode:n},o={x:r};return M.runKernel(hn,o,a)}var Uu=P({mirrorPad_:Hu});function Wu(e,t){let n=N(e,`a`,`mod`),r=N(t,`b`,`mod`);[n,r]=Gi(n,r);let i={a:n,b:r};return M.runKernel(`Mod`,i)}var Gu=P({mod_:Wu});function Ku(e,t=null,n=!1){e=N(e,`x`,`moments`);let r=O(t,e.shape),i=Lu(e,r,n),a=i.shape;return n||(a=il(i.shape,r)),{mean:i,variance:Lu(vl(W(I(e,`float32`),B(i,a))),r,n)}}var qu=P({moments_:Ku});function Ju(e,t){let n=N(e,`a`,`notEqual`,`string_or_numeric`),r=N(t,`b`,`notEqual`,`string_or_numeric`);[n,r]=Gi(n,r),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(vn,i)}var Yu=P({notEqual_:Ju});function Xu(e,t,n=1,r=0,i=`int32`){if(t<2)throw Error(`Error in oneHot: depth must be >=2, but it is ${t}`);let a={indices:N(e,`indices`,`oneHot`,`int32`)},o={dtype:i,depth:t,onValue:n,offValue:r};return M.runKernel(Cn,a,o)}var Zu=P({oneHot_:Xu});function Qu(e){let t={x:N(e,`x`,`onesLike`)};return M.runKernel(Sn,t)}var $u=P({onesLike_:Qu});function ed(e,t,n=0){let r=N(e,`x`,`pad`);if(r.rank===0)throw Error(`pad(scalar) is not defined. Pass non-scalar to pad`);let i={paddings:t,constantValue:n},a={x:r};return M.runKernel(Tn,a,i)}var td=P({pad_:ed});function nd(e,t,n){let r=N(e,`x`,`spaceToBatchND`);b(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),b(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),b(r.shape.reduce((e,r,i)=>i>0&&i<=t.length?e&&(r+n[i-1][0]+n[i-1][1])%t[i-1]===0:e,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);let i={x:r},a={blockShape:t,paddings:n};return M.runKernel(tr,i,a)}var rd=P({spaceToBatchND_:nd});function id(e,t,n,r,i,a,o){i??=[1,1],a??=1,r===0&&(r=`valid`);let s=N(e,`x`,`maxPool`),c=s,l=!1;s.rank===3&&(l=!0,c=B(s,[1,s.shape[0],s.shape[1],s.shape[2]])),b(ms(a,i),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${i}'`);let u=es(c.shape,t,a,i,r),d=[u.dilationHeight,u.dilationWidth],f;f=r===`same`?od([u.filterHeight,u.filterWidth],d):[[0,0],[0,0]];let p=d[0]===1&&d[1]===1,[m,h]=ad([u.inHeight,u.inWidth],d,f),g=p?r:`valid`,_=p?c:rd(c,d,m),v=(n===`avg`?()=>bs(_,t,a,g,o):()=>ju(_,t,a,g,o))(),y=p?v:Ps(v,d,h);return l?B(y,[y.shape[1],y.shape[2],y.shape[3]]):y}function ad(e,t,n){let r=n.map(e=>e[0]),i=n.map(e=>e[1]),a=e.concat(r,i),o=t.map((e,t)=>(e-a[t]%e)%e),s=i.map((e,t)=>e+o[t]);return[t.map((e,t)=>[r[t],s[t]]),t.map((e,t)=>[0,o[t]])]}function od(e,t){let n=e.map((e,n)=>e+(e-1)*(t[n]-1)).map(e=>e-1),r=n.map(e=>Math.floor(e/2)),i=n.map((e,t)=>e-r[t]);return n.map((e,t)=>[r[t],i[t]])}var sd=P({pool_:id});function cd(e,t){let n={x:N(e,`x`,`prelu`),alpha:N(t,`alpha`,`prelu`)};return M.runKernel(En,n)}var ld=P({prelu_:cd});function ud(e,t=null,n=!1){let r=N(e,`x`,`prod`);r.dtype===`bool`&&(r=I(r,`int32`));let i={x:r},a={axis:t,keepDims:n};return M.runKernel(Dn,i,a)}var dd=P({prod_:ud}),fd=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=o();t.next=function(){var e=2091639*t.s0+t.c*23283064365386963e-26;return t.s0=t.s1,t.s1=t.s2,t.s2=e-(t.c=e|0)},t.c=1,t.s0=n(` `),t.s1=n(` `),t.s2=n(` `),t.s0-=n(e),t.s0<0&&(t.s0+=1),t.s1-=n(e),t.s1<0&&(t.s1+=1),t.s2-=n(e),t.s2<0&&(t.s2+=1),n=null}function i(e,t){return t.c=e.c,t.s0=e.s0,t.s1=e.s1,t.s2=e.s2,t}function a(e,t){var n=new r(e),a=t&&t.state,o=n.next;return o.int32=function(){return n.next()*4294967296|0},o.double=function(){return o()+(o()*2097152|0)*11102230246251565e-32},o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}function o(){var e=4022871197;return function(t){t=String(t);for(var n=0;n<t.length;n++){e+=t.charCodeAt(n);var r=.02519603282416938*e;e=r>>>0,r-=e,r*=e,e=r>>>0,r-=e,e+=r*4294967296}return(e>>>0)*23283064365386963e-26}}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.alea=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),pd=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.x=0,t.y=0,t.z=0,t.w=0,t.next=function(){var e=t.x^t.x<<11;return t.x=t.y,t.y=t.z,t.z=t.w,t.w^=t.w>>>19^e^e>>>8},e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor128=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),md=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.x^t.x>>>2;return t.x=t.y,t.y=t.z,t.z=t.w,t.w=t.v,(t.d=t.d+362437|0)+(t.v=t.v^t.v<<4^(e^e<<1))|0},t.x=0,t.y=0,t.z=0,t.w=0,t.v=0,e===(e|0)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=n.charCodeAt(r)|0,r==n.length&&(t.d=t.x<<10^t.x>>>4),t.next()}function i(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t.v=e.v,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorwow=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),hd=o(((e,t)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.x,n=t.i,r=e[n],i;return r^=r>>>7,i=r^r<<24,r=e[n+1&7],i^=r^r>>>10,r=e[n+3&7],i^=r^r>>>3,r=e[n+4&7],i^=r^r<<7,r=e[n+7&7],r^=r<<13,i^=r^r<<9,e[n]=i,t.i=n+1&7,i};function n(e,t){var n,r=[];if(t===(t|0))r[0]=t;else for(t=``+t,n=0;n<t.length;++n)r[n&7]=r[n&7]<<15^t.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&r[n]===0;++n);for(n==8?r[7]=-1:r[n],e.x=r,e.i=0,n=256;n>0;--n)e.next()}n(t,e)}function i(e,t){return t.x=e.x.slice(),t.i=e.i,t}function a(e,t){e??=+new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.x&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xorshift7=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),gd=o(((e,t)=>{(function(e,t,n){function r(e){var t=this;t.next=function(){var e=t.w,n=t.X,r=t.i,i,a;return t.w=e=e+1640531527|0,a=n[r+34&127],i=n[r=r+1&127],a^=a<<13,i^=i<<17,a^=a>>>15,i^=i>>>12,a=n[r]=a^i,t.i=r,a+(e^e>>>16)|0};function n(e,t){var n,r,i,a,o,s=[],c=128;for(t===(t|0)?(r=t,t=null):(t+=`\0`,r=0,c=Math.max(c,t.length)),i=0,a=-32;a<c;++a)t&&(r^=t.charCodeAt((a+32)%t.length)),a===0&&(o=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,a>=0&&(o=o+1640531527|0,n=s[a&127]^=r+o,i=n==0?i+1:0);for(i>=128&&(s[(t&&t.length||0)&127]=-1),i=127,a=512;a>0;--a)r=s[i+34&127],n=s[i=i+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,s[i]=r^n;e.w=o,e.X=s,e.i=i}n(t,e)}function i(e,t){return t.i=e.i,t.w=e.w,t.X=e.X.slice(),t}function a(e,t){e??=+new Date;var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(a.X&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.xor4096=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),_d=o(((e,t)=>{(function(e,t,n){function r(e){var t=this,n=``;t.next=function(){var e=t.b,n=t.c,r=t.d,i=t.a;return e=e<<25^e>>>7^n,n=n-r|0,r=r<<24^r>>>8^i,i=i-e|0,t.b=e=e<<20^e>>>12^n,t.c=n=n-r|0,t.d=r<<16^n>>>16^i,t.a=i-e|0},t.a=0,t.b=0,t.c=-1640531527,t.d=1367130551,e===Math.floor(e)?(t.a=e/4294967296|0,t.b=e|0):n+=e;for(var r=0;r<n.length+20;r++)t.b^=n.charCodeAt(r)|0,t.next()}function i(e,t){return t.a=e.a,t.b=e.b,t.c=e.c,t.d=e.d,t}function a(e,t){var n=new r(e),a=t&&t.state,o=function(){return(n.next()>>>0)/4294967296};return o.double=function(){do var e=((n.next()>>>11)+(n.next()>>>0)/4294967296)/(1<<21);while(e===0);return e},o.int32=n.next,o.quick=o,a&&(typeof a==`object`&&i(a,n),o.state=function(){return i(n,{})}),o}t&&t.exports?t.exports=a:n&&n.amd?n(function(){return a}):this.tychei=a})(e,typeof t==`object`&&t,typeof define==`function`&&define)})),vd=o(((e,t)=>{(function(e,n,r){var i=256,a=6,o=52,s=`random`,c=r.pow(i,a),l=r.pow(2,o),u=l*2,d=i-1,f;function p(e,t,o){var d=[];t=t==1?{entropy:!0}:t||{};var f=_(g(t.entropy?[e,y(n)]:e??v(),3),d),p=new m(d),b=function(){for(var e=p.g(a),t=c,n=0;e<l;)e=(e+n)*i,t*=i,n=p.g(1);for(;e>=u;)e/=2,t/=2,n>>>=1;return(e+n)/t};return b.int32=function(){return p.g(4)|0},b.quick=function(){return p.g(4)/4294967296},b.double=b,_(y(p.S),n),(t.pass||o||function(e,t,n,i){return i&&(i.S&&h(i,p),e.state=function(){return h(p,{})}),n?(r[s]=e,t):e})(b,f,`global`in t?t.global:this==r,t.state)}function m(e){var t,n=e.length,r=this,a=0,o=r.i=r.j=0,s=r.S=[];for(n||(e=[n++]);a<i;)s[a]=a++;for(a=0;a<i;a++)s[a]=s[o=d&o+e[a%n]+(t=s[a])],s[o]=t;(r.g=function(e){for(var t,n=0,a=r.i,o=r.j,s=r.S;e--;)t=s[a=d&a+1],n=n*i+s[d&(s[a]=s[o=d&o+t])+(s[o]=t)];return r.i=a,r.j=o,n})(i)}function h(e,t){return t.i=e.i,t.j=e.j,t.S=e.S.slice(),t}function g(e,t){var n=[],r=typeof e,i;if(t&&r==`object`)for(i in e)try{n.push(g(e[i],t-1))}catch{}return n.length?n:r==`string`?e:e+`\0`}function _(e,t){for(var n=e+``,r,i=0;i<n.length;)t[d&i]=d&(r^=t[d&i]*19)+n.charCodeAt(i++);return y(t)}function v(){try{var t;return f&&(t=f.randomBytes)?t=t(i):(t=new Uint8Array(i),(e.crypto||e.msCrypto).getRandomValues(t)),y(t)}catch{var r=e.navigator,a=r&&r.plugins;return[+new Date,e,a,e.screen,y(n)]}}function y(e){return String.fromCharCode.apply(0,e)}if(_(r.random(),n),typeof t==`object`&&t.exports){t.exports=p;try{f=po()}catch{}}else typeof define==`function`&&define.amd?define(function(){return p}):r[`seed`+s]=p})(typeof self<`u`?self:e,[],Math)})),yd=l(o(((e,t)=>{var n=fd(),r=pd(),i=md(),a=hd(),o=gd(),s=_d(),c=vd();c.alea=n,c.xor128=r,c.xorwow=i,c.xorshift7=a,c.xor4096=o,c.tychei=s,t.exports=c}))()),bd=class{constructor(e,t,n,r,i){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);let a=i||Math.random();this.random=yd.alea(a.toString())}nextValue(){if(!isNaN(this.nextVal)){let e=this.nextVal;return this.nextVal=NaN,e}let e,t,n=!1;for(;!n;){let r,i,a;do r=2*this.random()-1,i=2*this.random()-1,a=r*r+i*i;while(a>=1||a===0);let o=Math.sqrt(-2*Math.log(a)/a);e=this.mean+this.stdDev*r*o,t=this.mean+this.stdDev*i*o,(!this.truncated||this.isValidTruncated(e))&&(n=!0)}return(!this.truncated||this.isValidTruncated(t))&&(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return this.dtype==null||this.dtype===`float32`?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}},xd=class{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype===`float32`,this.min=e,this.range=t-e,this.dtype=n,r??=Math.random(),typeof r==`number`&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=yd.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}};function Sd(e,t=0,n=1,r,i){if(Se(e),r!=null&&r===`bool`)throw Error(`Unsupported data type ${r}`);let a=new bd(t,n,r,!1,i),o=_o(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}var Cd=P({randomNormal_:Sd});function wd(e,t=0,n=1,r=`float32`,i){Se(e);let a=_o(e,r),o=new xd(t,n,null,i);for(let e=0;e<a.values.length;e++)a.values[e]=o.nextValue();return a.toTensor()}var Td=P({randomUniform_:wd});function Ed(e,t,n=1,r=`float32`){if(n===0)throw Error(`Cannot have a step of zero`);let i={start:e,stop:t,step:n,dtype:r};return M.runKernel(jn,{},i)}function Dd(e){let t={input:N(e,`input`,`real`)};return M.runKernel(Mn,t)}var Od=P({real_:Dd});function kd(e){let t={x:N(e,`x`,`reciprocal`)};return M.runKernel(Nn,t)}var Ad=P({reciprocal_:kd});function jd(e){let t={x:N(e,`x`,`relu`)};return M.runKernel(Pn,t)}var Md=P({relu_:jd});function Nd(e){let t={x:N(e,`x`,`relu6`)};return M.runKernel(Bn,t)}var Pd=P({relu6_:Nd});function Fd(e,t){let n={x:N(e,`x`,`reverse`)},r={dims:t};return M.runKernel(Vn,n,r)}var Id=P({reverse_:Fd});function Ld(e){let t={x:N(e,`x`,`round`)};return M.runKernel(Hn,t)}var Rd=P({round_:Ld});function zd(e){let t={x:N(e,`x`,`rsqrt`,`float32`)};return M.runKernel(Un,t)}var Bd=P({rsqrt_:zd});function Vd(e){let t={x:N(e,`x`,`selu`)};return M.runKernel(Jn,t)}var Hd=P({selu_:Vd});function Ud(e,t,n,r,i,a=[1,1],o=`NHWC`){let s=N(e,`x`,`separableConv2d`),c=N(t,`depthwiseFilter`,`separableConv2d`),l=N(n,`pointwiseFilter`,`separableConv2d`),u=s,d=!1;if(s.rank===3&&(d=!0,u=B(s,[1,s.shape[0],s.shape[1],s.shape[2]])),o===`NCHW`)throw Error(`separableConv2d currently does not support dataFormat NCHW; only NHWC is supported`);b(u.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${u.rank}.`),b(c.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${c.rank}.`),b(l.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${c.rank}.`),b(l.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${l.shape[0]}.`),b(l.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${l.shape[1]}.`);let f=c.shape[2],p=c.shape[3];b(l.shape[2]===f*p,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${f*p}, but got ${l.shape[2]}.`);let m=cc(Pc(u,c,r,i,o,a),l,1,`valid`,o);return d?B(m,[m.shape[1],m.shape[2],m.shape[3]]):m}var Wd=P({separableConv2d_:Ud});function Gd(e){let t={x:N(e,`x`,`sign`)};return M.runKernel(Zn,t)}var Kd=P({sign_:Gd});function qd(e){let t={x:N(e,`x`,`sin`,`float32`)};return M.runKernel(`Sin`,t)}var Jd=P({sin_:qd});function Yd(e){let t={x:N(e,`x`,`sinh`)};return M.runKernel(Xn,t)}var Xd=P({sinh_:Yd});function Zd(e,t,n){let r=N(e,`x`,`slice1d`);return b(r.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),As(r,[t],[n])}var Qd=P({slice1d_:Zd});function $d(e,t,n){let r=N(e,`x`,`slice2d`);return b(r.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),As(r,t,n)}var ef=P({slice2d_:$d});function tf(e,t,n){let r=N(e,`x`,`slice3d`);return b(r.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),As(r,t,n)}var nf=P({slice3d_:tf});function rf(e,t,n){let r=N(e,`x`,`slice4d`);return b(r.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),As(r,t,n)}var af=P({slice4d_:rf});function of(e,t=-1){let n=N(e,`logits`,`softmax`,`float32`);if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);let r={logits:n},i={dim:t};return M.runKernel(rr,r,i)}var sf=P({softmax_:of});function cf(e){b(e.dtype===`complex64`,()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);let t={input:e};return M.runKernel(`FFT`,t)}var lf=P({fft_:cf});function uf(e){b(e.dtype===`complex64`,()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);let t={input:e};return M.runKernel(Gt,t)}var df=P({ifft_:uf});function ff(e){let t=e.shape[e.shape.length-1],n=e.size/t,r;if(t<=2)r=df(B(e,[n,t]));else{let i=[n,2*(t-1)],a=B(Od(e),[n,t]),o=B(Wl(e),[n,t]),s=Id(As(a,[0,1],[n,t-2]),1),c=z(Id(As(o,[0,1],[n,t-2]),1),H(-1));r=df(B(fa(ws([a,s],1),ws([o,c],1)),[i[0],i[1]]))}if(r=Od(r),e.rank===3&&e.shape[0]!==0){let t=r,n=e.shape[0];r=B(r,[n,r.shape[0]/n,r.shape[1]]),t.dispose()}return r}var pf=P({irfft_:ff});function mf(e,t,n=0){let r={x:N(e,`x`,`split`)},i={numOrSizeSplits:t,axis:n};return M.runKernel(nr,r,i)}var hf=P({split_:mf});function gf(e,t){b(e.dtype===`float32`,()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1],r=e.size/n,i;if(t!=null&&t<n){let r=e.shape.map(e=>0),a=e.shape.map(e=>e);a[e.shape.length-1]=t,i=As(e,r,a),n=t}else if(t!=null&&t>n){let r=e.shape.map(e=>e);r[e.shape.length-1]=t-n,i=ws([e,Ru(r)],e.shape.length-1),n=t}else i=e;let a=Wc(i),o=lf(B(fa(i,a),[r,n])),s=Math.floor(n/2)+1,c=Od(o),l=Wl(o),u=hf(c,[s,n-s],c.shape.length-1),d=hf(l,[s,n-s],l.shape.length-1),f=i.shape.slice();return f[i.shape.length-1]=s,B(fa(u[0],d[0]),f)}var _f=P({rfft_:gf});function vf(e,t){let n=N(e,`a`,`squaredDifference`),r=N(t,`b`,`squaredDifference`);[n,r]=Gi(n,r),V(n.shape,r.shape);let i={a:n,b:r};return M.runKernel(lr,i,{})}var yf=P({squaredDifference_:vf});function bf(e,t){let n=N(e,`x`,`squeeze`,`string_or_numeric`);return B(n,ne(n.shape,t).newShape)}var xf=P({squeeze_:bf});function Sf(e,t=0){let n=la(e,`tensors`,`stack`,`string_or_numeric`);b(n.length>=1,()=>`Pass at least one tensor to tf.stack`),n.length>0&&b(t<=n[0].rank,()=>`Axis must be <= rank of the tensor`);let r=n,i={axis:t};return M.runKernel(wn,r,i)}var Cf=P({stack_:Sf});function wf(e,t=0){let n={x:N(e,`x`,`step`)},r={alpha:t};return M.runKernel(Tr,n,r)}var Tf=P({step_:wf});function Ef(e,t,n,r,i=0,a=0,o=0,s=0,c=0){let l={x:N(e,`x`,`stridedSlice`,`string_or_numeric`)},u={begin:t,end:n,strides:r,beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};return M.runKernel(fr,l,u)}var Df=P({stridedSlice_:Ef});function Of(e){let t={x:N(e,`x`,`tan`,`float32`)};return M.runKernel(`Tan`,t)}var kf=P({tan_:Of});function Af(e,t){S(e);let n=oa(e,t);if(n.length!==1)throw Error(`tensor1d() requires values to be a flat/TypedArray`);return pa(e,null,n,t)}function jf(e,t,n){if(S(e),t!=null&&t.length!==2)throw Error(`tensor2d() requires shape to have two numbers`);let r=oa(e,n);if(r.length!==2&&r.length!==1)throw Error(`tensor2d() requires values to be number[][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return pa(e,t,r,n)}function Mf(e,t,n){if(S(e),t!=null&&t.length!==4)throw Error(`tensor4d() requires shape to have four numbers`);let r=oa(e,n);if(r.length!==4&&r.length!==1)throw Error(`tensor4d() requires values to be number[][][][] or flat/TypedArray`);if(r.length===1&&t==null)throw Error("tensor4d() requires shape to be provided when `values` are a flat array");return pa(e,t,r,n)}function Nf(e,t,n){let r=t.rank>1?t.shape[t.rank-1]:1,i=t.rank>1?t.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${i}.`;if(n.rank<i)throw Error(a+` update.rank < ${i}. `);if(e.length<r+(n.rank-i))throw Error(a+` Output shape length < ${r+(n.rank-i)}`);if(n.rank!==i+e.length-r)throw Error(a+` update.rank != ${i+e.length-r}`);for(let e=0;e<i;++e)if(n.shape[e]!==t.shape[e])throw Error(a+` updates.shape[${e}] (${n.shape[e]}) != indices.shape[${e}] (${t.shape[e]}).`);for(let t=0;t<n.rank-i;++t)if(n.shape[t+i]!==e[t+r])throw Error(a+` updates.shape[${t+i}] (${n.shape[t+i]}) != shape[${t+i}] (${e[t+i]})`)}function Pf(e,t,n){if(t.rank<1)throw Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!==`int32`)throw Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Nf(n,t,e)}function Ff(e,t,n){let r=t.shape.length,i=r>1?t.shape[r-1]:1,a=n.length,o=1;for(let e=i;e<a;++e)o*=n[e];let s=i<1?1:i,c=C(t.shape)/s,l=[...k(n.slice(0,i)),1],u=C(n);return{sliceRank:i,numUpdates:c,sliceSize:o,strides:l,outputSize:u}}function If(e,t=1,n=!0){let r=N(e,`x`,`topk`);if(r.rank===0)throw Error(`topk() expects the input to be of rank 1 or higher`);let i=r.shape[r.shape.length-1];if(t<0)throw Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>i)throw Error(`'k' passed to topk() must be <= the last dimension (${i}) but got ${t}`);let a={x:r},o={k:t,sorted:n},[s,c]=M.runKernel(vr,a,o);return{values:s,indices:c}}var Lf=P({topk_:If});function Rf(e,t=0,n=1,r,i){if(Se(e),r!=null&&r===`bool`)throw Error(`Unsupported data type $ { dtype }`);let a=new bd(t,n,r,!0,i),o=_o(e,r);for(let e=0;e<o.values.length;e++)o.values[e]=a.nextValue();return o.toTensor()}var zf=P({truncatedNormal_:Rf});function Bf(e,t=0){let n=N(e,`x`,`unique`,`string_or_numeric`);b(n.rank>0,()=>`The input tensor must be at least 1D`);let r={x:n},i={axis:t},[a,o]=M.runKernel(xr,r,i);return{values:a,indices:o}}var Vf=P({unique_:Bf});function Hf(e,t,n){let r=N(e,`x`,`unsortedSegmentSum`),i=N(t,`segmentIds`,`unsortedSegmentSum`,`int32`);b(T(n),()=>`numSegments must be of dtype int`);let a={x:r,segmentIds:i},o={numSegments:n};return M.runKernel(Cr,a,o)}var Uf=P({unsortedSegmentSum_:Hf});function Wf(e,t=0){let n=N(e,`x`,`unstack`,`string_or_numeric`);b(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);let r={value:n},i={axis:t};return M.runKernel(Sr,r,i)}var Gf=P({unstack_:Wf});function Kf(e,t=!0,n,r){return M.makeVariable(e,t,n,r)}function qf(e,t){let n=[];for(let e=0;e<t.length;e++)t[e]&&n.push(e);let r=_o(e,`int32`),i=_o([n.length,e.length],`int32`);for(let t=0;t<n.length;t++){let a=r.indexToLoc(n[t]),o=t*e.length;i.values.set(a,o)}return i.toTensor()}function Jf(e,t,n){let r=N(e,`x`,`transpose`);if(t??=r.shape.map((e,t)=>t).reverse(),b(r.rank===t.length,()=>`Error in transpose: rank of input ${r.rank} must match length of perm ${t}.`),t.forEach(e=>{b(e>=0&&e<r.rank,()=>`All entries in 'perm' must be between 0 and ${r.rank-1} but got ${t}`)}),r.rank<=1)return r.clone();let i={x:r},a={perm:t};return r.dtype===`complex64`?F(()=>{let e=Od(r),t=Wl(r);return e=M.runKernel(br,{x:e},a),t=M.runKernel(br,{x:t},a),n&&(t=fu(t)),fa(e,t)}):M.runKernel(br,i,a)}var Yf=P({transpose_:Jf});function Xf(e,t){if(t==null)return e.shape.slice();if(w(e.shape,t))return t;if(e.shape.length===t.length){let n=[];for(let r=0;r<e.shape.length;r++)t[r]==null&&e.shape[r]!=null?n.push(e.shape[r]):n.push(t[r]);return n}return t}function Zf(e,t,n,r){let i=N(e,`x`,`dropout`);if(b(i.dtype===`float32`,()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${i.dtype} tensor instead.`),b(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof Ni?i.clone():i;let a=Xf(i,n),o=1-t;return z(i,R(Il(L(Td(a,0,1,`float32`,r),o)),o))}var Qf=P({dropout_:Zf});function $f(e,t,n,r,i,a=`NHWC`,o){let s=e;e.rank===3&&(s=B(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=B(t,[1,t.shape[0],t.shape[1],t.shape[2]])),b(s.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${s.shape}.`),b(c.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${c.shape}.`),b(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);let l=a===`NHWC`?s.shape[3]:s.shape[1],u=a===`NHWC`?c.shape[3]:c.shape[1];b(l===n[2],()=>`Error in conv2dDerFilter: depth of input ${l}) must match input depth in filter (${n[2]}.`),b(u===n[3],()=>`Error in conv2dDerFilter: depth of dy (${u}) must match output depth for filter (${n[3]}).`),_s(`conv2dDerFilter`,i,o);let d={x:s,dy:c},f={strides:r,pad:i,dataFormat:a,dimRoundingMode:o,filterShape:n};return M.runKernel(ut,d,f)}var ep=P({conv2DBackpropFilter_:$f});function tp(e,t,n){if(n==null||n===`linear`)return e;if(n===`relu`)return z(e,Tf(t));throw Error(`Cannot compute gradient for fused activation ${n}.`)}function np(e,t){let n=t,r=Rc(e.shape,t.shape);return r.length>0&&(n=U(n,r)),B(n,e.shape)}function rp(e,t,n,r){if(t===`linear`)return e;if(t===`relu`)return Md(e);if(t===`elu`)return Qc(e);if(t===`relu6`)return Pd(e);if(t===`prelu`)return ld(e,n);if(t===`leakyrelu`)return Ql(e,r);if(t===`sigmoid`)return Os(e);throw Error(`Unknown fused activation ${t}.`)}var ip=(e,t)=>!(e>0)||t===`linear`;function ap({x:e,filter:t,strides:n,pad:r,dataFormat:i=`NHWC`,dilations:a=[1,1],dimRoundingMode:o,bias:s,activation:c=`linear`,preluActivationWeights:l,leakyreluAlpha:u}){if(c||=`linear`,ip(M.state.gradientDepth,c)===!1){b(i===`NHWC`,()=>`Error in fused conv2d: got dataFormat of ${i} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let d=cc(e,t,n,r,i,a,o);return s!=null&&(d=L(d,s)),rp(d,c,l,u)}let d=N(e,`x`,`conv2d`,`float32`),f=N(t,`filter`,`conv2d`,`float32`),p=d,m=!1;d.rank===3&&(m=!0,p=B(d,[1,d.shape[0],d.shape[1],d.shape[2]])),b(p.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${p.rank}.`),b(f.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${f.rank}.`),_s(`fused conv2d`,r,o);let h=i===`NHWC`?p.shape[3]:p.shape[1];b(f.shape[2]===h,()=>`Error in conv2d: depth of input (${h}) must match input depth for filter ${f.shape[2]}.`),b(ms(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);let g=ns(p.shape,f.shape,n,a,r,o),_;s!=null&&(_=N(s,`bias`,`fused conv2d`),[_]=Gi(_,d),i===`NHWC`?V(g.outShape,_.shape):(b(_.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${_.shape.length}.`),b(_.shape.length===0||_.shape[0]===g.outChannels||_.shape[0]===1,()=>`Error in fused conv2d: bias shape (${_.shape}) is not compatible with the number of output channels (${g.outChannels})`)));let v;if(l!=null){let e=l.shape;if(b(e.length<=1||e.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${e.length}.`),e.length===1)b(e[0]===1||e[0]===g.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the number of output channels (${g.outChannels}).`);else if(e.length===3)try{V(e,g.outShape)}catch{let t=`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the output shape of the conv2d (${g.outShape}).`;throw Error(t)}v=N(l,`prelu weights`,`fused conv2d`)}let y=(e,t)=>{b(i===`NHWC`,()=>`Error in gradient of fused conv2D: got dataFormat of ${i} but only NHWC is currently supported.`);let[o,s,l,u]=t,d=tp(e,l,c);b(ps(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);let f=[fc(s.shape,d,o,n,r),ep(s,d,o.shape,n,r)];if(u!=null){let e=np(u,d);f.push(e)}return f},x={x:p,filter:f,bias:_,preluActivationWeights:v},S={strides:n,pad:r,dataFormat:i,dilations:a,dimRoundingMode:o,activation:c,leakyreluAlpha:u};return s==null?uu((e,t,n)=>{let r=M.runKernel(kr,x,S);return n([t,e,r]),m&&(r=B(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:y}})(p,f):uu((e,t,n,r)=>{let i=M.runKernel(kr,x,S);return r([t,e,i,n]),m&&(i=B(i,[i.shape[1],i.shape[2],i.shape[3]])),{value:i,gradFunc:y}})(p,f,_)}var op=P({fusedConv2d_:ap});function sp(e,t,n,r,i,a=[1,1],o){let s=e;e.rank===3&&(s=B(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=B(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let l={x:s,dy:c},u={strides:r,pad:i,dimRoundingMode:o,dilations:a,filterShape:n};return M.runKernel(St,l,u)}var cp=P({depthwiseConv2dNativeBackpropFilter_:sp});function lp(e,t,n,r,i,a=[1,1],o){let s=t,c=!1;t.rank===3&&(c=!0,s=B(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let l={dy:s,filter:n},u={strides:r,pad:i,dimRoundingMode:o,dilations:a,inputShape:e},d=M.runKernel(Ct,l,u);return c?B(d,[d.shape[1],d.shape[2],d.shape[3]]):d}var up=P({depthwiseConv2dNativeBackpropInput_:lp});function dp({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:i,activation:a=`linear`,preluActivationWeights:o,leakyreluAlpha:s=.2}){if(ip(M.state.gradientDepth,a)===!1){let c=Es(e,t,n,r);return i!=null&&(c=L(c,i)),rp(c,a,o,s)}let c=N(e,`a`,`fused matMul`),l=N(t,`b`,`fused matMul`);[c,l]=Gi(c,l);let u=n?c.shape[c.rank-2]:c.shape[c.rank-1],d=r?l.shape[l.rank-1]:l.shape[l.rank-2],f=n?c.shape[c.rank-1]:c.shape[c.rank-2],p=r?l.shape[l.rank-2]:l.shape[l.rank-1],m=c.shape.slice(0,-2),h=l.shape.slice(0,-2),g=C(m),_=C(h);b(u===d,()=>`Error in fused matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${c.shape} and ${l.shape} and transposeA=${n} and transposeB=${r} must match.`);let v=V(c.shape.slice(0,-2),l.shape.slice(0,-2)).concat([f,p]),y=n?B(c,[g,u,f]):B(c,[g,f,u]),x=r?B(l,[_,p,d]):B(l,[_,d,p]),S;i!=null&&(S=N(i,`bias`,`fused matMul`),[S]=Gi(S,c),V(v,S.shape));let w;o!=null&&(w=N(o,`prelu weights`,`fused matMul`));let T=(e,t)=>{let[o,s,c,l]=t,u=tp(B(e,c.shape),c,a),d,f;if(!n&&!r?(d=Es(u,s,!1,!0),f=Es(o,u,!0,!1)):!n&&r?(d=Es(u,s,!1,!1),f=Es(u,o,!0,!1)):n&&!r?(d=Es(s,u,!1,!0),f=Es(o,u,!1,!1)):(d=Es(s,u,!0,!0),f=Es(u,o,!0,!0)),i!=null){let e=np(l,u);return[d,f,e]}return[d,f]},E={a:y,b:x,bias:S,preluActivationWeights:w},D={transposeA:n,transposeB:r,activation:a,leakyreluAlpha:s};return i==null?uu((e,t,n)=>{let r=M.runKernel(Or,E,D);return n([e,t,r]),{value:B(r,v),gradFunc:T}})(y,x):uu((e,t,n,r)=>{let i=M.runKernel(Or,E,D);return r([e,t,i,n]),{value:B(i,v),gradFunc:T}})(y,x,S)}var fp=P({fusedMatMul_:dp});function pp(e,t,n,r,i=`bilinear`,a=0){let o=N(e,`image`,`cropAndResize`),s=N(t,`boxes`,`cropAndResize`,`float32`),c=N(n,`boxInd`,`cropAndResize`,`int32`),l=s.shape[0];b(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),b(s.rank===2&&s.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${l},4] but had shape ${s.shape}.`),b(c.rank===1&&c.shape[0]===l,()=>`Error in cropAndResize: boxInd must be have size [${l}] but had shape ${s.shape}.`),b(r.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),b(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),b(i===`bilinear`||i===`nearest`,()=>`method must be bilinear or nearest, but was ${i}`);let u={image:o,boxes:s,boxInd:c},d={method:i,extrapolationValue:a,cropSize:r};return M.runKernel(vt,u,d)}var mp=P({cropAndResize_:pp});function hp(e){let t=N(e,`image`,`flipLeftRight`,`float32`);b(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);let n={image:t};return M.runKernel(It,n,{})}var gp=P({flipLeftRight_:hp});function _p(e){let t=N(e,`image`,`grayscaleToRGB`),n=t.rank-1,r=t.shape[n];b(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),b(r===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);let i=Array(t.rank);return i.fill(1,0,n),i[n]=3,Ml(t,i)}var vp=P({grayscaleToRGB_:_p});function yp(e){let t=N(e,`image`,`RGBToGrayscale`),n=t.rank-1,r=t.shape[n];b(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),b(r===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${r}.`);let i=t.dtype,a=I(t,`float32`),o=Af([.2989,.587,.114]),s;switch(t.rank){case 2:s=Xc(`ij,j->i`,a,o);break;case 3:s=Xc(`ijk,k->ij`,a,o);break;case 4:s=Xc(`ijkl,l->ijk`,a,o);break;case 5:s=Xc(`ijklm,m->ijkl`,a,o);break;case 6:s=Xc(`ijklmn,n->ijklm`,a,o);break;default:throw Error(`Not a valid tensor rank.`)}return s=Ol(s,-1),I(s,i)}var bp=P({rgbToGrayscale_:yp});function xp(e,t,n=0,r=.5){let i=N(e,`image`,`rotateWithOffset`,`float32`);b(i.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${i.rank}.`);let a={image:i},o={radians:t,fillValue:n,center:r};return M.runKernel(Dr,a,o)}var Sp=P({rotateWithOffset_:xp});function Cp(e,t,n,r,i,a){r??=.5,i??=-1/0,a??=0;let o=e.shape[0];return n=Math.min(n,o),b(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),b(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),b(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),b(t.rank===1,()=>`scores must be a 1D tensor`),b(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),b(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:i,softNmsSigma:a}}function wp(e,t,n,r=.5,i=-1/0){let a=N(e,`boxes`,`nonMaxSuppression`,`float32`),o=N(t,`scores`,`nonMaxSuppression`,`float32`),s=Cp(a,o,n,r,i);n=s.maxOutputSize,r=s.iouThreshold,i=s.scoreThreshold;let c={maxOutputSize:n,iouThreshold:r,scoreThreshold:i};return M.runKernel(yn,{boxes:a,scores:o},c)}var Tp=P({nonMaxSuppression_:wp});function Ep(e,t,n){let r=Dp(e,t,n),i=r<0?-(r+1):r;e.splice(i,0,t)}function Dp(e,t,n){return kp(e,t,n||Op)}function Op(e,t){return e>t?1:e<t?-1:0}function kp(e,t,n){let r=0,i=e.length,a=0,o=!1;for(;r<i;){a=r+(i-r>>>1);let s=n(t,e[a]);s>0?r=a+1:(i=a,o=!s)}return o?r:-r-1}function Ap(e,t,n,r,i){return Np(e,t,n,r,i,0)}function jp(e,t,n,r,i,a){return Np(e,t,n,r,i,0,!1,a,!0)}function Mp(e,t,n,r,i,a){return Np(e,t,n,r,i,a,!0)}function Np(e,t,n,r,i,a,o=!1,s=!1,c=!1){let l=[];for(let e=0;e<t.length;e++)t[e]>i&&l.push({score:t[e],boxIndex:e,suppressBeginIndex:0});l.sort(Ip);let u=a>0?-.5/a:0,d=[],f=[];for(;d.length<n&&l.length>0;){let t=l.pop(),{score:n,boxIndex:a,suppressBeginIndex:o}=t;if(n<i)break;let s=!1;for(let n=d.length-1;n>=o;--n){let o=Pp(e,a,d[n]);if(o>=r){s=!0;break}if(t.score*=Fp(r,u,o),t.score<=i)break}t.suppressBeginIndex=d.length,s||(t.score===n?(d.push(a),f.push(t.score)):t.score>i&&Ep(l,t,Ip))}let p=d.length,m=n-p;s&&m>0&&(d.push(...Array(m).fill(0)),f.push(...Array(m).fill(0)));let h={selectedIndices:d};return o&&(h.selectedScores=f),c&&(h.validOutputs=p),h}function Pp(e,t,n){let r=e.subarray(t*4,t*4+4),i=e.subarray(n*4,n*4+4),a=Math.min(r[0],r[2]),o=Math.min(r[1],r[3]),s=Math.max(r[0],r[2]),c=Math.max(r[1],r[3]),l=Math.min(i[0],i[2]),u=Math.min(i[1],i[3]),d=Math.max(i[0],i[2]),f=Math.max(i[1],i[3]),p=(s-a)*(c-o),m=(d-l)*(f-u);if(p<=0||m<=0)return 0;let h=Math.max(a,l),g=Math.max(o,u),_=Math.min(s,d),v=Math.min(c,f),y=Math.max(_-h,0)*Math.max(v-g,0);return y/(p+m-y)}function Fp(e,t,n){let r=Math.exp(t*n*n);return n<=e?r:0}function Ip(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}async function Lp(e,t,n,r=.5,i=-1/0){let a=N(e,`boxes`,`nonMaxSuppressionAsync`),o=N(t,`scores`,`nonMaxSuppressionAsync`),s=Cp(a,o,n,r,i);n=s.maxOutputSize,r=s.iouThreshold,i=s.scoreThreshold;let c=await Promise.all([a.data(),o.data()]),l=c[0],u=c[1],{selectedIndices:d}=Ap(l,u,n,r,i);return a!==e&&a.dispose(),o!==t&&o.dispose(),Af(d,`int32`)}var Rp=Lp;function zp(e,t,n,r=.5,i=-1/0,a=0){let o=N(e,`boxes`,`nonMaxSuppression`),s=N(t,`scores`,`nonMaxSuppression`),c=Cp(o,s,n,r,i,a);n=c.maxOutputSize,r=c.iouThreshold,i=c.scoreThreshold,a=c.softNmsSigma;let l={boxes:o,scores:s},u={maxOutputSize:n,iouThreshold:r,scoreThreshold:i,softNmsSigma:a},d=M.runKernel(xn,l,u);return{selectedIndices:d[0],selectedScores:d[1]}}var Bp=P({nonMaxSuppressionWithScore_:zp});async function Vp(e,t,n,r=.5,i=-1/0,a=0){let o=N(e,`boxes`,`nonMaxSuppressionAsync`),s=N(t,`scores`,`nonMaxSuppressionAsync`),c=Cp(o,s,n,r,i,a);n=c.maxOutputSize,r=c.iouThreshold,i=c.scoreThreshold,a=c.softNmsSigma;let l=await Promise.all([o.data(),s.data()]),u=l[0],d=l[1],{selectedIndices:f,selectedScores:p}=Mp(u,d,n,r,i,a);return o!==e&&o.dispose(),s!==t&&s.dispose(),{selectedIndices:Af(f,`int32`),selectedScores:Af(p)}}var Hp=Vp;function Up(e,t,n,r=.5,i=-1/0,a=!1){let o=N(e,`boxes`,`nonMaxSuppression`),s=N(t,`scores`,`nonMaxSuppression`),c=Cp(o,s,n,r,i,null),l=c.maxOutputSize,u=c.iouThreshold,d=c.scoreThreshold,f={boxes:o,scores:s},p={maxOutputSize:l,iouThreshold:u,scoreThreshold:d,padToMaxOutputSize:a},m=M.runKernel(bn,f,p);return{selectedIndices:m[0],validOutputs:m[1]}}var Wp=P({nonMaxSuppressionPadded_:Up});async function Gp(e,t,n,r=.5,i=-1/0,a=!1){let o=N(e,`boxes`,`nonMaxSuppressionAsync`),s=N(t,`scores`,`nonMaxSuppressionAsync`),c=Cp(o,s,n,r,i,null),l=c.maxOutputSize,u=c.iouThreshold,d=c.scoreThreshold,[f,p]=await Promise.all([o.data(),s.data()]),{selectedIndices:m,validOutputs:h}=jp(f,p,l,u,d,a);return o!==e&&o.dispose(),s!==t&&s.dispose(),{selectedIndices:Af(m,`int32`),validOutputs:H(h,`int32`)}}var Kp=Gp;function qp(e,t,n=!1,r=!1){let i=N(e,`images`,`resizeBilinear`);b(i.rank===3||i.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${i.rank}.`),b(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),b(r===!1||n===!1,()=>`Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.`);let a=i,o=!1;i.rank===3&&(o=!0,a=B(i,[1,i.shape[0],i.shape[1],i.shape[2]]));let[]=t,s={images:a},c={alignCorners:n,halfPixelCenters:r,size:t},l=M.runKernel(Rn,s,c);return o?B(l,[l.shape[1],l.shape[2],l.shape[3]]):l}var Jp=P({resizeBilinear_:qp});function Yp(e,t,n=!1,r=!1){let i=N(e,`images`,`resizeNearestNeighbor`);b(i.rank===3||i.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${i.rank}.`),b(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),b(i.dtype===`float32`||i.dtype===`int32`,()=>"`images` must have `int32` or `float32` as dtype"),b(r===!1||n===!1,()=>`Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.`);let a=i,o=!1;i.rank===3&&(o=!0,a=B(i,[1,i.shape[0],i.shape[1],i.shape[2]]));let[]=t,s={images:a},c={alignCorners:n,halfPixelCenters:r,size:t},l=M.runKernel(In,s,c);return o?B(l,[l.shape[1],l.shape[2],l.shape[3]]):l}var Xp=P({resizeNearestNeighbor_:Yp});function Zp(e,t=`binary`,n=!1,r=.5){let i=N(e,`image`,`threshold`),a=i.shape[0]*i.shape[1],o=z(Af([r]),255),s,c,l,u;if(b(i.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${i.rank}.`),b(i.shape[2]===3||i.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${i.shape[2]}.`),b(i.dtype===`int32`||i.dtype===`float32`,()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${i.dtype}.`),b(t===`otsu`||t===`binary`,()=>`Method must be binary or otsu, but was ${t}`),i.shape[2]===3){[s,c,l]=hf(i,[1,1,1],-1);let e=z(s,.2989),t=z(c,.587),n=z(l,.114);u=L(L(e,t),n)}else u=e;return t===`otsu`&&(o=Qp(Gs(I(Rd(u),`int32`),ma([]),256),a)),I(z(n?nu(u,o):Bl(u,o),255),`int32`)}function Qp(e,t){let n=Af([-1]),r=Af([0]),i=Af([0]),a,o,s,c,l,u;for(let d=0;d<e.size-1;d++){a=As(e,0,d+1),o=As(e,d+1),l=R(U(a),t),u=R(U(o),t),s=R(U(z(a,Ed(0,a.size))),U(a));let f=Xs(o.shape,a.size),p=L(Ed(0,o.size),f);c=R(U(z(o,p)),U(o));let m=W(s,c),h=W(s,c);i=z(z(z(l,u),m),h);let g=Bl(i,r);r=Hc(g,i,r),n=Hc(g,Af([d]),n)}return n}var $p=P({threshold_:Zp});function em(e,t,n=`nearest`,r=`constant`,i=0,a){let o=N(e,`image`,`transform`,`float32`),s=N(t,`transforms`,`transform`,`float32`);b(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),b(s.rank===2&&(s.shape[0]===o.shape[0]||s.shape[0]===1)&&s.shape[1]===8,()=>`Error in transform: Input transform should be batch x 8 or 1 x 8`),b(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);let c={image:o,transforms:s},l={interpolation:n,fillMode:r,fillValue:i,outputShape:a};return M.runKernel(yr,c,l)}var tm=P({transform_:em});function nm(e,t,n){let r=N(e,`a`,`bandPart`);b(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);let i=r.shape,[a,o]=r.shape.slice(-2),s,c;typeof t==`number`?(b(t%1==0,()=>`bandPart(): numLower must be an integer, got ${t}.`),b(t<=a,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${a}).`),s=N(t<0?a:t,`numLower`,`bandPart`)):(b(t.dtype===`int32`,()=>`bandPart(): numLower's dtype must be an int32.`),s=Hc(eu(t,0),a,Vu(t,a))),typeof n==`number`?(b(n%1==0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),b(n<=o,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`),c=N(n<0?o:n,`numUpper`,`bandPart`)):(b(n.dtype===`int32`,()=>`bandPart(): numUpper's dtype must be an int32.`),c=Hc(eu(n,0),o,Vu(n,o)));let l=W(B(Ed(0,a,1,`int32`),[-1,1]),Ed(0,o,1,`int32`)),u=Cu(nu(l,s),Hl(l,fu(c))),d=Ru([a,o],r.dtype);return B(Cf(Gf(B(r,[-1,a,o])).map(e=>Hc(u,e,d))),i)}var rm=P({bandPart_:nm});function im(e){let t;if(Array.isArray(e)){t=!1,b(e!=null&&e.length>0,()=>`Gram-Schmidt process: input must not be null, undefined, or empty`);let n=e[0].shape[0];for(let t=1;t<e.length;++t)b(e[t].shape[0]===n,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[t].shape[0]} vs. ${n})`)}else t=!0,e=hf(e,e.shape[0],0).map(e=>xf(e,[0]));b(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);let n=[],r=e;for(let t=0;t<e.length;++t)n.push(M.tidy(()=>{let e=r[t];if(t>0)for(let r=0;r<t;++r){let t=z(U(z(n[r],e)),n[r]);e=W(e,t)}return R(e,Sl(e,`euclidean`))}));return t?Cf(n,0):n}var am=P({gramSchmidt_:im});function om(e,t=!1){if(b(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return sm(e,t);{let n=Gf(B(e,[e.shape.slice(0,e.shape.length-2).reduce((e,t)=>e*t),e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),r=[],i=[];return n.forEach(e=>{let[n,a]=sm(e,t);r.push(n),i.push(a)}),[B(Cf(r,0),e.shape),B(Cf(i,0),e.shape)]}}function sm(e,t=!1){return M.tidy(()=>{b(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);let n=e.shape[0],r=e.shape[1],i=Pl(n),a=bo(e),o=jf([[1]],[1,1]),s=bo(o),c=n>=r?r:n;for(let e=0;e<c;++e){let t=a,c=s,l=i;[s,a,i]=M.tidy(()=>{let t=As(a,[e,e],[n-e,1]),c=Sl(t),l=As(a,[e,e],[1,1]),u=Hc(Bl(l,0),jf([[-1]]),jf([[1]])),d=W(l,z(u,c)),f=R(t,d);s=f.shape[0]===1?bo(o):ws([o,As(f,[1,0],[f.shape[0]-1,f.shape[1]])],0);let p=fu(R(Es(u,d),c)),m=As(a,[e,0],[n-e,r]),h=z(p,s),g=Yf(s);if(e===0)a=W(m,Es(h,Es(g,m)));else{let t=W(m,Es(h,Es(g,m)));a=ws([As(a,[0,0],[e,r]),t],0)}let _=Yf(h),v=As(i,[0,e],[n,i.shape[1]-e]);if(e===0)i=W(v,Es(Es(v,s),_));else{let t=W(v,Es(Es(v,s),_));i=ws([As(i,[0,0],[n,e]),t],1)}return[s,a,i]}),ya([t,c,l])}return!t&&n>r&&(i=As(i,[0,0],[n,r]),a=As(a,[0,0],[r,r])),[i,a]})}var cm=P({qr_:om}),lm;(function(e){e[e.NONE=0]=`NONE`,e[e.MEAN=1]=`MEAN`,e[e.SUM=2]=`SUM`,e[e.SUM_BY_NONZERO_WEIGHTS=3]=`SUM_BY_NONZERO_WEIGHTS`})(lm||={});function um(e,t,n=lm.SUM_BY_NONZERO_WEIGHTS){let r=N(e,`losses`,`computeWeightedLoss`),i=null;t!=null&&(i=N(t,`weights`,`computeWeightedLoss`));let a=i==null?r:z(r,i);if(n===lm.NONE)return a;if(n===lm.SUM)return U(a);if(n===lm.MEAN){if(i==null)return Lu(a);{let e=r.size/i.size,t=R(U(a),U(i));return e>1?R(t,H(e)):t}}if(n===lm.SUM_BY_NONZERO_WEIGHTS){if(i==null)return R(U(a),H(r.size));{let e=I(U(Yu(z(i,zu(r.shape)),H(0))),`float32`);return R(U(a),e)}}throw Error(`Unknown reduction: ${n}`)}var dm=P({computeWeightedLoss_:um});function fm(e,t,n,r=lm.SUM_BY_NONZERO_WEIGHTS){let i=N(e,`labels`,`absoluteDifference`),a=N(t,`predictions`,`absoluteDifference`),o=null;return n!=null&&(o=N(n,`weights`,`absoluteDifference`)),x(i.shape,a.shape,`Error in absoluteDifference: `),dm(Oo(W(i,a)),o,r)}var pm=P({absoluteDifference_:fm});function mm(e,t,n,r,i=lm.SUM_BY_NONZERO_WEIGHTS){let a=N(e,`labels`,`cosineDistance`),o=N(t,`predictions`,`cosineDistance`),s=null;return r!=null&&(s=N(r,`weights`,`cosineDistance`)),x(a.shape,o.shape,`Error in cosineDistance: `),dm(W(H(1),U(z(a,o),n,!0)),s,i)}var hm=P({cosineDistance_:mm});function gm(e,t,n,r=lm.SUM_BY_NONZERO_WEIGHTS){let i=N(e,`labels`,`hingeLoss`),a=N(t,`predictions`,`hingeLoss`),o=null;n!=null&&(o=N(n,`weights`,`hingeLoss`)),x(i.shape,a.shape,`Error in hingeLoss: `);let s=H(1);return i=W(z(H(2),i),s),dm(Md(W(s,z(i,a))),o,r)}var _m=P({hingeLoss_:gm});function vm(e,t,n,r=1,i=lm.SUM_BY_NONZERO_WEIGHTS){let a=N(e,`labels`,`huberLoss`),o=N(t,`predictions`,`huberLoss`),s=null;n!=null&&(s=N(n,`weights`,`huberLoss`)),x(a.shape,o.shape,`Error in huberLoss: `);let c=H(r),l=Oo(W(o,a)),u=Vu(l,c),d=W(l,u);return dm(L(z(H(.5),vl(u)),z(c,d)),s,i)}var ym=P({huberLoss_:vm});function bm(e,t,n,r=1e-7,i=lm.SUM_BY_NONZERO_WEIGHTS){let a=N(e,`labels`,`logLoss`),o=N(t,`predictions`,`logLoss`),s=null;n!=null&&(s=N(n,`weights`,`logLoss`)),x(a.shape,o.shape,`Error in logLoss: `);let c=H(1),l=H(r);return dm(W(fu(z(a,ou(L(o,l)))),z(W(c,a),ou(L(W(c,o),l)))),s,i)}var xm=P({logLoss_:bm});function Sm(e,t,n,r=lm.SUM_BY_NONZERO_WEIGHTS){let i=N(e,`labels`,`meanSquaredError`),a=N(t,`predictions`,`meanSquaredError`),o=null;return n!=null&&(o=N(n,`weights`,`meanSquaredError`)),x(i.shape,a.shape,`Error in meanSquaredError: `),dm(yf(i,a),o,r)}var Cm=P({meanSquaredError_:Sm});function wm(e,t){let n=N(e,`labels`,`sigmoidCrossEntropyWithLogits`),r=N(t,`logits`,`sigmoidCrossEntropyWithLogits`);x(n.shape,r.shape,`Error in sigmoidCrossEntropyWithLogits: `);let i=Md(r),a=z(r,n),o=cu(El(fu(Oo(r))));return L(W(i,a),o)}function Tm(e,t,n,r=0,i=lm.SUM_BY_NONZERO_WEIGHTS){let a=N(e,`multiClassLabels`,`sigmoidCrossEntropy`),o=N(t,`logits`,`sigmoidCrossEntropy`),s=null;if(n!=null&&(s=N(n,`weights`,`sigmoidCrossEntropy`)),x(a.shape,o.shape,`Error in sigmoidCrossEntropy: `),r>0){let e=H(r),t=H(1),n=H(.5);a=L(z(a,W(t,e)),z(n,e))}return dm(wm(a,o),s,i)}var Em=P({sigmoidCrossEntropy_:Tm});function Dm(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return uu((e,t,r)=>{let i=xu(t,[n],!0),a=W(I(t,`float32`),i);return r([e,a]),{value:U(fu(z(a,e)),[n]),gradFunc:(e,t)=>{let[r,i]=t,a=il(e.shape,[n]);return[z(B(e,a),W(I(r,`float32`),El(i))),z(B(e,a),W(El(i),I(r,`float32`)))]}}})(e,t)}function Om(e,t,n,r=0,i=lm.SUM_BY_NONZERO_WEIGHTS){let a=N(e,`onehotLabels`,`softmaxCrossEntropy`),o=N(t,`logits`,`softmaxCrossEntropy`),s=null;if(n!=null&&(s=N(n,`weights`,`softmaxCrossEntropy`)),x(a.shape,o.shape,`Error in softmaxCrossEntropy: `),r>0){let e=H(r),t=H(1),n=H(a.shape[1]);a=L(z(a,W(t,e)),R(e,n))}return dm(Dm(a,o),s,i)}var km=P({softmaxCrossEntropy_:Om}),Am={flipLeftRight:gp,grayscaleToRGB:vp,resizeNearestNeighbor:Xp,resizeBilinear:Jp,rgbToGrayscale:bp,rotateWithOffset:Sp,cropAndResize:mp,nonMaxSuppression:Tp,nonMaxSuppressionAsync:Rp,nonMaxSuppressionWithScore:Bp,nonMaxSuppressionWithScoreAsync:Hp,nonMaxSuppressionPadded:Wp,nonMaxSuppressionPaddedAsync:Kp,threshold:$p,transform:tm},jm={bandPart:rm,gramSchmidt:am,qr:cm},Mm={absoluteDifference:pm,computeWeightedLoss:dm,cosineDistance:hm,hingeLoss:_m,huberLoss:ym,logLoss:xm,meanSquaredError:Cm,sigmoidCrossEntropy:Em,softmaxCrossEntropy:km},Nm=new Map,Pm=new Map,Fm=class{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}},Im=class e{constructor(){this.classNameMap={}}static getMap(){return e.instance??=new e,e.instance}static register(t){e.getMap().classNameMap[t.className]=[t,t.fromConfig]}};function G(e,t,n){b(e.className!=null,()=>`Class being registered does not have the static className property defined.`),b(typeof e.className==`string`,()=>`className is required to be a string, but got type `+typeof e.className),b(e.className.length>0,()=>`Class being registered has an empty-string as its className, which is disallowed.`),t===void 0&&(t=`Custom`),n===void 0&&(n=e.className);let r=n,i=t+`>`+r;return Im.register(e),Nm.set(i,e),Pm.set(e,i),e}var Lm=class extends Fm{minimize(e,t=!1,n){let{value:r,grads:i}=this.computeGradients(e,n);if(n!=null){let e=n.map(e=>({name:e.name,tensor:i[e.name]}));this.applyGradients(e)}else this.applyGradients(i);return ya(i),t?r:(r.dispose(),null)}get iterations(){return this.iterations_??=0,this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return lu(e,t)}dispose(){this.iterations_!=null&&ya(this.iterations_)}async saveIterations(){return this.iterations_??=0,{name:`iter`,tensor:H(this.iterations_,`int32`)}}async getWeights(){throw Error(`getWeights() is not implemented for this optimizer yet.`)}async setWeights(e){throw Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}};Object.defineProperty(Lm,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});var Rm=class extends Lm{static get className(){return`Adadelta`}constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],n??(this.epsilon=M.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=M.registeredVariables[t];this.accumulatedGrads[n]??(this.accumulatedGrads[n]={originalName:`${t}/accum_grad`,variable:F(()=>Wc(r).variable(!1))}),this.accumulatedUpdates[n]??(this.accumulatedUpdates[n]={originalName:`${t}/accum_var`,variable:F(()=>Wc(r).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedGrads[n].variable,o=this.accumulatedUpdates[n].variable;F(()=>{let e=L(z(a,this.rho),z(vl(i),1-this.rho)),t=z(R(gl(L(o,this.epsilon)),gl(L(a,this.epsilon))),i),n=L(z(o,this.rho),z(vl(t),1-this.rho));a.assign(e),o.assign(n);let s=L(z(t,-this.learningRate),r);r.assign(s)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(ya(this.accumulatedGrads.map(e=>e.variable)),ya(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){let e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=e.length/2;this.accumulatedGrads=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedUpdates=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}},zm=class extends Lm{static get className(){return`Adagrad`}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=M.registeredVariables[t];this.accumulatedGrads[n]??(this.accumulatedGrads[n]={originalName:`${t}/accumulator`,variable:F(()=>Xs(r.shape,this.initialAccumulatorValue).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedGrads[n].variable;F(()=>{let e=L(a,vl(i));a.assign(e);let t=L(z(R(i,gl(L(e,M.backend.epsilon()))),-this.learningRate),r);r.assign(t)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&ya(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulatedGrads=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}},Bm=class extends Lm{static get className(){return`Adam`}constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],F(()=>{this.accBeta1=H(t).variable(),this.accBeta2=H(n).variable()}),r??(this.epsilon=M.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);F(()=>{let n=W(1,this.accBeta1),r=W(1,this.accBeta2);t.forEach((t,i)=>{let a=M.registeredVariables[t];this.accumulatedFirstMoment[i]??(this.accumulatedFirstMoment[i]={originalName:`${t}/m`,variable:F(()=>Wc(a).variable(!1))}),this.accumulatedSecondMoment[i]??(this.accumulatedSecondMoment[i]={originalName:`${t}/v`,variable:F(()=>Wc(a).variable(!1))});let o=Array.isArray(e)?e[i].tensor:e[t];if(o==null)return;let s=this.accumulatedFirstMoment[i].variable,c=this.accumulatedSecondMoment[i].variable,l=L(z(s,this.beta1),z(o,1-this.beta1)),u=L(z(c,this.beta2),z(vl(o),1-this.beta2)),d=R(l,n),f=R(u,r);s.assign(l),c.assign(u);let p=L(z(R(d,L(gl(f),this.epsilon)),-this.learningRate),a);a.assign(p)}),this.accBeta1.assign(z(this.accBeta1,this.beta1)),this.accBeta2.assign(z(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&ya(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&ya(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),F(()=>{this.accBeta1.assign(ml(this.beta1,this.iterations_+1)),this.accBeta2.assign(ml(this.beta2,this.iterations_+1))});let t=e.length/2;this.accumulatedFirstMoment=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedSecondMoment=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}},Vm=class extends Lm{static get className(){return`Adamax`}constructor(e,t,n,r=null,i=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=i,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],F(()=>{this.iteration=H(0).variable(),this.accBeta1=H(t).variable()}),r??(this.epsilon=M.backend.epsilon())}applyGradients(e){let t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);F(()=>{let n=W(1,this.accBeta1),r=R(-this.learningRate,L(z(this.iteration,this.decay),1));t.forEach((t,i)=>{let a=M.registeredVariables[t];this.accumulatedFirstMoment[i]??(this.accumulatedFirstMoment[i]={originalName:`${t}/m`,variable:Wc(a).variable(!1)}),this.accumulatedWeightedInfNorm[i]??(this.accumulatedWeightedInfNorm[i]={originalName:`${t}/v`,variable:Wc(a).variable(!1)});let o=Array.isArray(e)?e[i].tensor:e[t];if(o==null)return;let s=this.accumulatedFirstMoment[i].variable,c=this.accumulatedWeightedInfNorm[i].variable,l=L(z(s,this.beta1),z(o,1-this.beta1)),u=Fu(z(c,this.beta2),Oo(o));s.assign(l),c.assign(u);let d=L(z(R(r,n),R(l,L(u,this.epsilon))),a);a.assign(d)}),this.iteration.assign(L(this.iteration,1)),this.accBeta1.assign(z(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&ya(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&ya(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw Error(`getWeights() is not implemented for Adamax yet.`)}async setWeights(e){throw Error(`setWeights() is not implemented for Adamax yet.`)}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}},Hm=class extends Lm{static get className(){return`SGD`}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=Array.isArray(e)?e[n].tensor:e[t];if(r==null)return;let i=M.registeredVariables[t];F(()=>{let e=L(z(this.c,r),i);i.assign(e)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=ba(H(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw Error(`SGD optimizer does not have settable weights.`)}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}},Um=class extends Hm{static get className(){return`Momentum`}constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=H(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=M.registeredVariables[t];this.accumulations[n]??(this.accumulations[n]={originalName:`${t}/momentum`,variable:F(()=>Wc(r).variable(!1))});let i=this.accumulations[n].variable,a=Array.isArray(e)?e[n].tensor:e[t];a!=null&&F(()=>{let e,t=L(z(this.m,i),a);e=this.useNesterov?L(z(this.c,L(a,z(t,this.m))),r):L(z(this.c,t),r),i.assign(t),r.assign(e)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&ya(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),this.accumulations=e.map(e=>({originalName:e.name,variable:e.tensor.variable(!1)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}},Wm=class extends Lm{static get className(){return`RMSProp`}constructor(e,t=.9,n=0,r=null,i=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=i,r??(this.epsilon=M.backend.epsilon()),e==null)throw Error(`learningRate for RMSPropOptimizer must be defined.`)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{let r=M.registeredVariables[t];this.accumulatedMeanSquares[n]??(this.accumulatedMeanSquares[n]={originalName:`${t}/rms`,variable:F(()=>Wc(r).variable(!1))}),this.accumulatedMoments[n]??(this.accumulatedMoments[n]={originalName:`${t}/momentum`,variable:F(()=>Wc(r).variable(!1))}),this.accumulatedMeanGrads[n]==null&&this.centered&&(this.accumulatedMeanGrads[n]={originalName:`${t}/mg`,variable:F(()=>Wc(r).variable(!1))});let i=Array.isArray(e)?e[n].tensor:e[t];if(i==null)return;let a=this.accumulatedMeanSquares[n].variable,o=this.accumulatedMoments[n].variable;F(()=>{let e=L(z(a,this.decay),z(vl(i),1-this.decay));if(this.centered){let t=this.accumulatedMeanGrads[n].variable,s=L(z(t,this.decay),z(i,1-this.decay)),c=R(z(i,this.learningRate),gl(W(e,L(vl(s),this.epsilon)))),l=L(z(o,this.momentum),c);a.assign(e),t.assign(s),o.assign(l);let u=W(r,l);r.assign(u)}else{let e=L(z(a,this.decay),z(vl(i),1-this.decay)),t=L(z(o,this.momentum),R(z(i,this.learningRate),gl(L(e,this.epsilon))));a.assign(e),o.assign(t);let n=W(r,t);r.assign(n)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&ya(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&ya(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&ya(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){let e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);let t=this.centered?e.length/3:e.length/2;this.accumulatedMeanSquares=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.accumulatedMoments=e.slice(t,t*2).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})),this.centered&&(this.accumulatedMeanGrads=e.slice(t*2,t*3).map(e=>({originalName:e.name,variable:e.tensor.variable(!1)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}},Gm=[Rm,zm,Bm,Vm,Um,Wm,Hm];function Km(){for(let e of Gm)G(e)}function qm(e,t){let n=e.shape.length,r=t.shape.length;if(n<1)throw Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if(t.dtype!==`int32`)throw Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(C(e.shape)===0)throw Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);let i=t.shape,a=i[i.length-1],o=1;for(let e=0;e<i.length-1;++e)o*=i[e];let s=e.shape,c=i.slice();c.pop();let l=1;for(let e=a;e<n;++e)l*=s[e],c.push(s[e]);let u=[...k(e.shape).map(e=>e/l),1].slice(0,a);return[c,o,l,u]}var Jm=s({assertParamsValid:()=>Zm,computeFlatOffset:()=>uh,computeOutShape:()=>$m,getNormalizedAxes:()=>rh,isSliceContinous:()=>lh,maskToAxes:()=>Qm,parseSliceParams:()=>dh,sliceInfo:()=>fh,startForAxis:()=>sh,startIndicesWithElidedDims:()=>ih,stopForAxis:()=>ch,stopIndicesWithElidedDims:()=>ah,stridesForAxis:()=>oh,stridesWithElidedDims:()=>eh}),Ym=-2,Xm=-1;function Zm(e,t,n){let r=e.shape.length;b(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),b(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let i=0;i<r;++i)b(t[i]+n[i]<=e.shape[i],()=>`Error in slice${r}D: begin[${i}] + size[${i}] (${t[i]+n[i]}) would overflow input.shape[${i}] (${e.shape[i]})`)}function Qm(e){let t=[],n=0;for(;e>0;)e&1&&t.push(n),e/=2,n++;return t}function $m(e,t,n){let r=[];for(let i=0;i<e.length;i++)r[i]=Math.ceil((t[i]-e[i])/n[i]);return r}function eh(e,t,n,r){let i=[...e];for(let e=i.length;e<r.length;e++)i.push(1);for(let e=0;e<n;e++)e===0?i[t]=1:(i.splice(t,0,1),i.pop());return i}function th(e,t,n){return n<=e?n:n-(t-1)}function nh(e,t){let n=[];for(let r=0;r<e;r++)n.push(t+r);return n}function rh(e,t,n,r,i,a,o,s,c){let l=e.length,u=Array(l),d=Array(l),f=Array(l);if(t.length&&n>0){let c=t[0],l=n+1;u=ih(o,c,l,r,e),d=ah(s,c,l,i,e),f=eh(a,c,l,e)}else for(let t=0;t<l;t++)u[t]=sh(o,r,a,e,t,c),d[t]=ch(s,i,a,e,t,c),f[t]=oh(a,t,c);return{begin:u,end:d,strides:f}}function ih(e,t,n,r,i){let a=[...i],o=nh(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=0;else{let o=th(t,n,i),s=r[o];e&1<<o&&(s=0),a[i]=s}return a}function ah(e,t,n,r,i){let a=[...i],o=nh(n,t);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=2**53-1;else{let o=th(t,n,i),s=r[o];e&1<<o&&(s=2**53-1),a[i]=s}for(let e=0;e<a.length;e++){let t=i[e];a[e]<0&&(a[e]+=t),a[e]=g(0,a[e],i[e])}return a}function oh(e,t,n){let r=e[t];return(n&1<<t||r==null)&&(r=1),r}function sh(e,t,n,r,i,a){let o=t[i],s=n[i]||1;(e&1<<i||a&1<<i||o==null)&&(o=s>0?-(2**53-1):2**53-1);let c=r[i];return o<0&&(o+=c),o=g(0,o,c-1),o}function ch(e,t,n,r,i,a){let o=t[i],s=n[i]||1;(e&1<<i||a&1<<i||o==null)&&(o=s>0?2**53-1:-(2**53-1));let c=r[i];return o<0&&(o+=c),o=s>0?g(0,o,c):g(-1,o,c-1),o}function lh(e,t,n){let r=n.length;for(let e=0;e<n.length;e++)if(n[e]>1){r=e;break}for(let i=r+1;i<n.length;i++)if(t[i]>0||n[i]!==e[i])return!1;return!0}function uh(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function dh(e,t,n){let r,i=e.shape.length;r=typeof t==`number`?[t,...Array(i-1).fill(0)]:t.length<i?t.concat(Array(i-t.length).fill(0)):t.slice(),r.forEach(e=>{b(e!==-1,()=>`slice() does not support negative begin indexing.`)});let a;return a=n==null?Array(i).fill(-1):typeof n==`number`?[n,...Array(i-1).fill(-1)]:n.length<i?n.concat(Array(i-n.length).fill(-1)):n,a=a.map((t,n)=>t>=0?t:(b(t===-1,()=>`Negative size values should be exactly -1 but got ${t} for the slice() size at index ${n}.`),e.shape[n]-r[n])),[r,a]}function fh(e,t,n,r,i,a,o,s,c){let l;if(r==null?(l=Array(t.length),l.fill(1)):l=r,o!=null&&o&o-1)throw Error(`Multiple ellipses in slice is not allowed.`);let u=!1,d={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:i,endMask:a,ellipsisMask:o,newAxisMask:s,shrinkAxisMask:c};for(let e=0;e<d.dims;e++)u&&1<<e&s&&d.numAddAxisAfterEllipsis++,1<<e&o&&(u=!0);u||(d.ellipsisMask|=1<<d.dims,d.dims++);let f={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};ph(d,f);let p=!0,m=!0,h=!0,g=[],_=[];for(let t=0;t<e.length;++t){if(f.strides[t]===0)throw Error(`strides[${t}] must be non-zero`);let n=!!(f.shrinkAxisMask&1<<t),r=e[t];if(r===-1){g.push(n?1:-1);continue}let i=[f.beginMask&1<<t,f.endMask&1<<t],a=[f.strides[t]>0?0:-1,f.strides[t]>0?r:r-1];if(n&&f.strides[t]<=0)throw Error(`only stride 1 allowed on non-range indexing.`);h&&=f.strides[t]===1;let o=!!(f.beginMask&1<<t&&f.endMask&1<<t);if(f.beginValid&&f.endValid){if(n){let e=f.begin[t]<0?r+f.begin[t]:f.begin[t];if(f.begin[t]=e,f.end[t]=f.begin[t]+1,e<0||e>=r)throw Error(`slice index ${f.begin[t]} of dimension ${t} out of bounds.`)}else f.begin[t]=mh(f.begin[t],0,f.strides[t],r,i,a),f.end[t]=mh(f.end[t],1,f.strides[t],r,i,a);let e=f.strides[t]===1&&f.begin[t]===0&&f.end[t]===r;p&&=e,m&&=t===0&&f.strides[t]===1||e}else p=p&&f.strides[t]===1&&o,m&&=t===0&&f.strides[t]===1||o;let s,c=!1;if(f.beginValid&&f.endValid?(s=f.end[t]-f.begin[t],c=!0):n?(s=1,c=!0):o&&r>=0&&(s=f.strides[t]<0?-r:r,c=!0),c){let e;e=s===0||s<0!=f.strides[t]<0?0:Math.trunc(s/f.strides[t])+(s%f.strides[t]===0?0:1),g.push(e)}else g.push(-1)}for(let e=0;e<f.finalShapeGatherIndices.length;++e){let t=f.finalShapeGatherIndices[e];t>=0?_.push(g[t]):t===Ym&&_.push(1)}return{finalShapeSparse:_.filter((e,t)=>f.finalShapeGatherIndices[t]!==Ym),finalShape:_,isIdentity:p,sliceDim0:m,isSimpleSlice:h,begin:f.begin,end:f.end,strides:f.strides}}function ph(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=Array(t.dims),t.end=Array(t.dims),t.strides=Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){let i=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<i;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(Ym),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[r]),e.end!=null&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(Xm),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}function mh(e,t,n,r,i,a){if(i[t])return n>0?a[t]:a[t+1&1];{let t=e<0?r+e:e;return t<a[0]?a[0]:t>a[1]?a[1]:t}}var hh=class{static sgd(e){return new Hm(e)}static momentum(e,t,n=!1){return new Um(e,t,n)}static rmsprop(e,t=.9,n=0,r=null,i=!1){return new Wm(e,t,n,r,i)}static adam(e=.001,t=.9,n=.999,r=null){return new Bm(e,t,n,r)}static adadelta(e=.001,t=.95,n=null){return new Rm(e,t,n)}static adamax(e=.002,t=.9,n=.999,r=null,i=0){return new Vm(e,t,n,r,i)}static adagrad(e,t=.1){return new zm(e,t)}},gh=typeof requestAnimationFrame<`u`?requestAnimationFrame:typeof setImmediate<`u`?setImmediate:e=>e();function _h(){return new Promise(e=>gh(()=>e()))}function vh(e,t){let n=e[0].length;e.forEach((e,t)=>{b(e.length===n,()=>`Error in concat${n}D: rank of tensors[${t}] must be the same as the rank of the rest (${n})`)}),b(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);let r=e[0];e.forEach((e,i)=>{for(let a=0;a<n;a++)b(a===t||e[a]===r[a],()=>`Error in concat${n}D: Shape of tensors[${i}] (${e}) does not match the shape of the rest (${r}) along the non-concatenated axis ${i}.`)})}function yh(e,t){let n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}var bh;(function(e){e[e.FIRST_DIM_SIZE=0]=`FIRST_DIM_SIZE`,e[e.VALUE_ROWIDS=1]=`VALUE_ROWIDS`,e[e.ROW_LENGTHS=2]=`ROW_LENGTHS`,e[e.ROW_SPLITS=3]=`ROW_SPLITS`,e[e.ROW_LIMITS=4]=`ROW_LIMITS`,e[e.ROW_STARTS=5]=`ROW_STARTS`})(bh||={});function xh(e,t,n){let r=[];if(n==null&&t==null)return r;if(t==null)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(n==null)return r;if(e+n.length!==r.length)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let i=1;i<n.length;++i){let a=n[i],o=r[r.length-n.length+i],s=r[o];if(a>=0){if(s>=0){if(s!==a)throw Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${i+e}] = ${a} but shape[${i+e}] = ${s}`)}else r[o]=a}}return r}function Sh(e){let t={FIRST_DIM_SIZE:bh.FIRST_DIM_SIZE,VALUE_ROWIDS:bh.VALUE_ROWIDS,ROW_LENGTHS:bh.ROW_LENGTHS,ROW_SPLITS:bh.ROW_SPLITS,ROW_LIMITS:bh.ROW_LIMITS,ROW_STARTS:bh.ROW_STARTS},n=[];for(let r of e)if(r in t)n.push(t[r]);else break;return n}function Ch(e){return e.length===0?0:e[0]===bh.FIRST_DIM_SIZE?e.length-1:e.length}function wh(e,t){if(e==null||t==null)return;let n=e.length,r=t.length;if(n>=r)throw Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let i=0;i<Math.min(n,r-1);++i){let n=e[i],r=t[i+1];if(n>=0&&r>=0&&n!==1&&n!==r)throw Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${i-e.length}] = ${n} but ragged tensor input.flatValues.shape[${i-e.length}] = ${r}`)}}function Th(e){return e<=30?e:he(e,Math.floor(Math.sqrt(e)))}function Eh(e,t,n){return[n*(typeof e==`number`?e:e[0]),t*(typeof e==`number`?e:e[1])]}function Dh(e,t,n,r=!0){let i=[];if(r)i=i.concat(t.slice(0)),i.push(e[0]/n),i=i.concat(e.slice(1));else{i=i.concat(e[0]);let n=t.length;for(let r=0;r<n;++r)i=i.concat([e[r+1]/t[r],t[r]]);i=i.concat(e.slice(n+1))}return i}function Oh(e,t,n=!0){let r=[];if(n){r.push(t);for(let n=t+1;n<e;++n)n<=2*t?(r.push(n),r.push(n-(t+1))):r.push(n)}else{let n=[],i=[];for(let r=1;r<e;++r)r>=t*2+1||r%2==1?i.push(r):n.push(r);r.push(...n),r.push(0),r.push(...i)}return r}function kh(e,t,n,r=!0){let i=[];r?i.push(e[0]/n):i.push(e[0]*n);for(let n=1;n<e.length;++n)n<=t.length?r?i.push(t[n-1]*e[n]):i.push(e[n]/t[n-1]):i.push(e[n]);return i}function Ah(e,t){let n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function jh(e,t,n){let r=e.slice(0,1);for(let i=0;i<n;++i)r.push(e[i+1]-t[i][0]-t[i][1]);return r}var Mh=1.7580993408473768,Nh=1.0507009873554805,Ph=.3275911,Fh=.254829592,Ih=-.284496736,Lh=1.421413741,Rh=-1.453152027,zh=1.061405429;function Bh(e,t){if(e.length!==t.length)throw Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);let n=new Float32Array(e.length*2);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}function Vh(e){let t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}}function Hh(e){let t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=0;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function Uh(e){let t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let t=2;t<e.length;t+=4)n[Math.floor(t/4)]=e[t],r[Math.floor(t/4)]=e[t+1];return{real:n,imag:r}}function Wh(e,t){return{real:e[t*2],imag:e[t*2+1]}}function Gh(e,t,n,r){e[r*2]=t,e[r*2+1]=n}function Kh(e,t){let n=new Float32Array(e/2),r=new Float32Array(e/2);for(let i=0;i<Math.ceil(e/2);i++){let a=(t?2:-2)*Math.PI*(i/e);n[i]=Math.cos(a),r[i]=Math.sin(a)}return{real:n,imag:r}}function qh(e,t,n){let r=(n?2:-2)*Math.PI*(e/t);return{real:Math.cos(r),imag:Math.sin(r)}}var Jh=`->`,Yh=/->/g,Xh=`,`,Zh=`...`;function Qh(e,t){e=e.replace(/\s/g,``);let n=(e.length-e.replace(Yh,``).length)/2;if(n<1)throw Error(`Equations without an arrow are not supported.`);if(n>1)throw Error(`Equation must contain exactly one arrow ("${Jh}").`);let[r,i]=e.split(Jh);b(r.indexOf(Zh)===-1,()=>`The ellipsis notation ("${Zh}") is not supported yet.`);let a=r.split(Xh),o=a.length;if(t!==o)throw Error(`Expected ${o} input tensors, received ${t}`);if(o>2)throw Error(`Support for more than 2 input tensors is not implemented yet.`);let s=[];for(let e=0;e<i.length;++e){let t=i[e];if(!a.some(e=>e.indexOf(t)!==-1))throw Error(`Output subscripts contain the label ${t} not present in the input subscripts.`);s.indexOf(t)===-1&&s.push(t)}for(let e=0;e<r.length;++e){let t=r[e];s.indexOf(t)===-1&&t!==Xh&&s.push(t)}let c=Array(a.length);for(let e=0;e<o;++e){if(new Set(a[e].split(``)).size!==a[e].length)throw Error(`Found duplicate axes in input component ${a[e]}. Support for duplicate axes in input is not implemented yet.`);c[e]=[];for(let t=0;t<a[e].length;++t)c[e].push(s.indexOf(a[e][t]))}let l=s.length,u=i.length,d=[];for(let e=u;e<l;++e)d.push(e);return{allDims:s,summedDims:d,idDims:c}}function $h(e,t){let n=Array(e);n.fill(-1);for(let e=0;e<t.length;++e)n[t[e]]=e;let r=[];for(let t=0;t<e;++t)n[t]===-1&&r.push(t);return n=n.filter(e=>e!==-1),{permutationIndices:n,expandDims:r}}function eg(e,t,n){let r=Array(e);for(let e=0;e<n.length;++e){let i=n[e].shape;for(let n=0;n<t[e].length;++n)r[t[e][n]]===void 0?r[t[e][n]]=i[n]:b(r[t[e][n]]===i[n],()=>`Expected dimension ${r[t[e][n]]} at axis ${n} of input shaped ${JSON.stringify(i)}, but got dimension ${i[n]}`)}}function tg(e,t){let n=e,r=[],i=0;e.length===0&&n.push(-1),i=e.length+1;for(let e=0;e<i;++e)r.push([]);let a=[];for(let e=0;e<n.length;++e){let i=n[e],o=rg(t,i);for(let t of o)a.indexOf(t)===-1&&(r[e].push(t),a.push(t))}return{path:n,steps:r}}function ng(e){return e.every((e,t)=>e===t)}function rg(e,t){let n=[];for(let r=0;r<e.length;++r)(e[r].length===0||e[r].indexOf(t)!==-1||t===-1)&&n.push(r);return n}function ig(e,t,n=0){let r=[];if(typeof t==`number`)b(e.shape[n]%t===0,()=>`Number of splits must evenly divide the axis.`),r=Array(t).fill(e.shape[n]/t);else{b(t.reduce((e,t)=>(t===-1&&(e+=1),e),0)<=1,()=>`There should be only one negative value in split array.`);let i=t.indexOf(-1);if(i!==-1){let r=t.reduce((e,t)=>t>0?e+t:e);t[i]=e.shape[n]-r}b(e.shape[n]===t.reduce((e,t)=>e+t),()=>`The sum of sizes must match the size of the axis dimension.`),r=t}return r}function ag(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function og(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function sg(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}function cg(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function lg(e,t){return`size ${e} must be non-negative, not ${t}`}function ug(){return`reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero`}function dg(e,t){return`Input to reshape is a SparseTensor with ${C(e)}
  dense values, but the requested shape requires a multiple of ${C(t)}. inputShape=${e} outputShape= ${t}`}function fg(e,t){return`Input to reshape is a tensor with ${C(e)} dense values, but the requested shape has ${C(t)}. inputShape=${e} outputShape=${t}`}function pg(){return`segment ids must be >= 0`}function mg(){return`segment ids are not increasing`}function hg(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function gg(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}var _g=s({collectGatherOpShapeInfo:()=>bg,computeOutShape:()=>yg,segOpComputeOptimalWindowSize:()=>vg});function vg(e,t){let n=!1,r;for(e<=30?(r=e,n=!0):r=he(e,Math.floor(Math.sqrt(e)));!n;)r>t||r===e?n=!0:r=he(e,r+1);return r}function yg(e,t,n){let r=[],i=e.length;for(let a=0;a<i;a++)a===t?r.push(n):r.push(e[a]);return r}function bg(e,t,n,r){let i=t.shape.length,a=e.shape.length;if(r!==0&&(r<-i||r>i))throw Error(`Expect batchDims in the range of [-${i}, ${i}], but got ${r}`);if(r<0&&(r+=i),r>a)throw Error(`batchDims (${r}) must be less than rank(x) (
    ${a}).`);if(n<r)throw Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let n=0;n<r;++n)if(e.shape[n]!==t.shape[n])throw Error(`x.shape[${n}]: ${e.shape[n]} should be equal to indices.shape[${n}]: ${t.shape[n]}.`);let o=e.shape[n],s=[],c=1,l=1,u=1;for(let t=0;t<r;++t)s.push(e.shape[t]),c*=e.shape[t];for(let t=r;t<n;t++)s.push(e.shape[t]),l*=e.shape[t];for(let e=r;e<i;e++)s.push(t.shape[e]);for(let t=n+1;t<a;t++)s.push(e.shape[t]),u*=e.shape[t];return{batchSize:c,sliceSize:u,outerSize:l,dimSize:o,outputShape:s}}var xg=s({ERF_A1:()=>Fh,ERF_A2:()=>Ih,ERF_A3:()=>Lh,ERF_A4:()=>Rh,ERF_A5:()=>zh,ERF_P:()=>Ph,PARALLELIZE_THRESHOLD:()=>30,RowPartitionType:()=>bh,SELU_SCALE:()=>Nh,SELU_SCALEALPHA:()=>Mh,applyActivation:()=>rp,assertAndGetBroadcastShape:()=>V,assertAxesAreInnerMostDims:()=>al,assertParamsConsistent:()=>vh,assignToTypedArray:()=>Gh,axesAreInnerMostDims:()=>tl,calculateShapes:()=>Ff,checkEinsumDimSizes:()=>eg,checkPadOnDimRoundingMode:()=>_s,combineLocations:()=>nl,combineRaggedTensorToTensorShapes:()=>xh,complexWithEvenIndex:()=>Hh,complexWithOddIndex:()=>Uh,computeConv2DInfo:()=>ns,computeConv3DInfo:()=>rs,computeDefaultPad:()=>os,computeDilation2DInfo:()=>$o,computeOptimalWindowSize:()=>Th,computeOutAndReduceShapes:()=>rl,computeOutShape:()=>yh,computePool2DInfo:()=>es,computePool3DInfo:()=>ts,convertConv2DDataFormat:()=>gs,decodeEinsumEquation:()=>Qh,eitherStridesOrDilationsAreOne:()=>ms,expandShapeToKeepDim:()=>il,exponent:()=>qh,exponents:()=>Kh,fromStringArrayToUint8:()=>Cg,fromUint8ToStringArray:()=>Sg,getAxesPermutation:()=>ol,getBroadcastDims:()=>Lc,getComplexWithIndex:()=>Wh,getEinsumComputePath:()=>tg,getEinsumPermutation:()=>$h,getFusedBiasGradient:()=>np,getFusedDyActivation:()=>tp,getImageCenter:()=>Eh,getInnerMostAxes:()=>cl,getPermuted:()=>Oh,getRaggedRank:()=>Ch,getReductionAxes:()=>Rc,getReshaped:()=>Dh,getReshapedPermuted:()=>kh,getRowPartitionTypesHelper:()=>Sh,getSliceBeginCoords:()=>Ah,getSliceSize:()=>jh,getSparseFillEmptyRowsIndicesDenseShapeMismatch:()=>ag,getSparseFillEmptyRowsNegativeIndexErrorMessage:()=>og,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:()=>sg,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:()=>ug,getSparseReshapeInputOutputMismatchErrorMessage:()=>fg,getSparseReshapeInputOutputMultipleErrorMessage:()=>dg,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:()=>cg,getSparseReshapeNegativeOutputDimErrorMessage:()=>lg,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:()=>gg,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:()=>pg,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:()=>mg,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:()=>hg,getUndoAxesPermutation:()=>sl,isIdentityPermutation:()=>ng,log:()=>Mr,mergeRealAndImagArrays:()=>Bh,prepareAndValidate:()=>qm,prepareSplitSize:()=>ig,segment_util:()=>_g,shouldFuse:()=>ip,slice_util:()=>Jm,splitRealAndImagArrays:()=>Vh,stridesOrDilationsArePositive:()=>hs,tupleValuesAreOne:()=>ps,upcastType:()=>Vi,validateDefaultValueShape:()=>wh,validateInput:()=>Pf,validateUpdateShape:()=>Nf,warn:()=>jr});function Sg(e){try{return e.map(e=>di(e))}catch(e){throw Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function Cg(e){return e.map(e=>ui(e))}Km();var wg={kernelName:`Abs`,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(e,Tf(I(n,`float32`),-1))}}},Tg={kernelName:Le,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>{let t=vl(I(n,`float32`));return fu(R(e,gl(W(H(1),t))))}}}},Eg={kernelName:Re,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,gl(W(vl(I(n,`float32`)),1)))}}},Dg={kernelName:`Add`,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t,i=V(n.shape,r.shape);return{a:()=>{let t=e,r=Rc(n.shape,i);return r.length>0&&(t=U(t,r)),B(t,n.shape)},b:()=>{let t=e,n=Rc(r.shape,i);return n.length>0&&(t=U(t,n)),B(t,r.shape)}}}},Og={kernelName:ze,saveAllInputs:!0,gradFunc:(e,t)=>{let n={};return t.forEach((t,r)=>{n[r]=()=>e.clone()}),n}},kg={kernelName:Be,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>Wc(n)}}},Ag={kernelName:Ve,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>Wc(n)}}},jg={kernelName:He,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,gl(W(H(1),vl(I(n,`float32`)))))}}},Mg={kernelName:Ue,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,gl(L(H(1),vl(I(n,`float32`)))))}}},Ng={kernelName:Ke,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t,i=V(n.shape,r.shape);return{a:()=>{let t=L(vl(n),vl(r)),a=z(e,R(r,t)),o=Rc(n.shape,i);return o.length>0&&(a=U(a,o)),B(a,n.shape)},b:()=>{let t=L(vl(n),vl(r)),a=fu(z(e,R(n,t))),o=Rc(r.shape,i);return o.length>0&&(a=U(a,o)),B(a,r.shape)}}}},Pg={kernelName:We,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,L(vl(I(n,`float32`)),1))}}},Fg={kernelName:Ge,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,W(H(1),vl(I(n,`float32`))))}}};function Ig(e,t,n,r,i,a){let o=N(e,`dy`,`avgPool3dGrad`),s=N(t,`input`,`avgPool3dGrad`),c=o,l=s,u=!1;s.rank===4&&(u=!0,c=B(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]]),l=B(s,[1,s.shape[0],s.shape[1],s.shape[2],s.shape[3]])),b(c.rank===5,()=>`Error in avgPool3dGrad: dy must be rank 5 but got rank ${c.rank}.`),b(l.rank===5,()=>`Error in avgPool3dGrad: input must be rank 5 but got rank ${l.rank}.`),_s(`avgPool3dGrad`,i,a);let d={dy:c,input:l},f={filterSize:n,strides:r,pad:i,dimRoundingMode:a},p=M.runKernel(Xe,d,f);return u?B(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}var Lg=P({avgPool3dGrad_:Ig}),Rg={kernelName:Ye,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{filterSize:i,strides:a,pad:o,dimRoundingMode:s}=n;return{x:()=>Lg(e,r,i,a,o,s)}}};function zg(e,t,n,r,i){let a=N(e,`dy`,`avgPoolGrad`),o=N(t,`input`,`avgPoolGrad`);b(o.rank===a.rank,()=>`Rank of input (${o.rank}) does not match rank of dy (${a.rank})`);let s=o,c=a,l=!1;o.rank===3&&(l=!0,s=B(o,[1,o.shape[0],o.shape[1],o.shape[2]]),c=B(a,[1,a.shape[0],a.shape[1],a.shape[2]])),b(c.rank===4,()=>`Error in avgPoolGrad: dy must be rank 4 but got rank ${c.rank}.`),b(s.rank===4,()=>`Error in avgPoolGrad: input must be rank 4 but got rank ${s.rank}.`);let u={dy:c,input:s},d={filterSize:n,strides:r,pad:i},f=M.runKernel(Je,u,d);return l?B(f,[f.shape[1],f.shape[2],f.shape[3]]):f}var Bg=P({avgPoolGrad_:zg}),Vg={kernelName:qe,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{filterSize:i,strides:a,pad:o}=n;return{x:()=>Bg(e,r,i,a,o)}}},Hg={kernelName:Ze,inputsToSave:[`a`,`b`],gradFunc:(e,t,n)=>{let[r,i]=t,{transposeA:a,transposeB:o}=n;return!a&&!o?{a:()=>Es(e,i,!1,!0),b:()=>Es(r,e,!0,!1)}:!a&&o?{a:()=>Es(e,i,!1,!1),b:()=>Es(e,r,!0,!1)}:a&&!o?{a:()=>Es(i,e,!1,!0),b:()=>Es(r,e,!1,!1)}:{a:()=>Es(i,e,!0,!0),b:()=>Es(e,r,!0,!0)}}},Ug={kernelName:Qe,gradFunc:(e,t,n)=>{let{blockShape:r,crops:i}=n;return{x:()=>rd(e,r,i)}}},Wg={kernelName:tt,gradFunc:(e,t,n)=>{let r=n,i=r.inputShape,a=r.shape,o=Array.from(a);for(let e=i.length-1;e>=0;e--)if(i[e]===a[e])o[e]=1;else if(i[e]!==1)throw Error(`broadcastTo(): [${i}] cannot be broadcast to [${a}].`);let s=[];for(let e=0;e<o.length;e++)o[e]>1&&s.push(e);return{x:()=>U(e,s,!0)}}},Gg={kernelName:rt,gradFunc:e=>({x:()=>e.clone()})},Kg={kernelName:it,gradFunc:e=>({x:()=>Wc(e)})},qg={kernelName:at,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{clipValueMin:i,clipValueMax:a}=n;return{x:()=>Hc(Cu(Hl(r,i),nu(r,a)),e,Wc(e))}}},Jg={kernelName:st,inputsToSave:[`x`],gradFunc:wg.gradFunc},Yg={kernelName:ct,saveAllInputs:!0,gradFunc:(e,t,n)=>{let r=t.map(e=>e.shape),{axis:i}=n,a=O(i,t[0].shape)[0];return hf(e,r.map(e=>e[a]),a).map(e=>()=>e)}},Xg={kernelName:lt,inputsToSave:[`x`,`filter`],gradFunc:(e,t,n)=>{let[r,i]=t,{dilations:a,strides:o,pad:s,dataFormat:c}=n;return b(ps(a),()=>`Error in gradient of conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`),{x:()=>fc(r.shape,e,i,o,s,c),filter:()=>ep(r,e,i.shape,o,s,c)}}},Zg={kernelName:dt,inputsToSave:[`dy`,`filter`],gradFunc:(e,t,n)=>{let[r,i]=t,{strides:a,pad:o,dataFormat:s,dimRoundingMode:c}=n;return{dy:()=>cc(e,i,a,o,s,1,c),filter:()=>ep(e,r,i.shape,a,o,s,c)}}};function Qg(e,t,n,r,i){let a=e;e.rank===4&&(a=B(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]]));let o=t;o.rank===4&&(o=B(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]])),b(a.rank===5,()=>`Error in conv3dDerFilter: input must be rank 5, but got shape ${a.shape}.`),b(o.rank===5,()=>`Error in conv3dDerFilter: dy must be rank 5, but got shape ${o.shape}.`),b(n.length===5,()=>`Error in conv3dDerFilter: filterShape must be length 5, but got ${n}.`),b(a.shape[4]===n[3],()=>`Error in conv3dDerFilter: depth of input ${a.shape[4]}) must match input depth in filter (${n[3]}.`),b(o.shape[4]===n[4],()=>`Error in conv3dDerFilter: depth of dy (${o.shape[4]}) must match output depth for filter (${n[4]}).`);let s={x:a,dy:o},c={strides:r,pad:i,filterShape:n};return M.runKernel(pt,s,c)}var $g=P({conv3DBackpropFilter_:Qg}),e_={kernelName:ft,inputsToSave:[`x`,`filter`],gradFunc:(e,t,n)=>{let{dilations:r,strides:i,pad:a}=n;b(ps(r),()=>`Error in gradient of conv3D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${r}'`);let[o,s]=t;return{x:()=>vc(o.shape,e,s,i,a),filter:()=>$g(o,e,s.shape,i,a)}}},t_={kernelName:`Cos`,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(fu(Jd(I(n,`float32`))),e)}}},n_={kernelName:ht,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(Xd(I(n,`float32`)),e)}}},r_={kernelName:_t,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{axis:i,exclusive:a,reverse:o}=n;return{x:()=>{let t=ol([i],r.rank),n=Oc(e,i,a,!o);return t!=null&&(n=Yf(n,t)),n}}}},i_={kernelName:xt,inputsToSave:[`x`,`filter`],gradFunc:(e,t,n)=>{let{dilations:r,strides:i,pad:a,dimRoundingMode:o}=n,s=r??[1,1];b(ps(s),()=>`Error in gradient of depthwiseConv2dNative: dilation rates greater than 1 are not yet supported. Got dilations '${s}'`);let[c,l]=t;return b(c.rank===4,()=>`Error in gradient of depthwiseConv2dNative: input must be rank 4, but got rank ${c.rank}.`),b(l.rank===4,()=>`Error in gradient of depthwiseConv2dNative: filter must be rank 4, but got rank ${l.rank}.`),b(c.shape[3]===l.shape[2],()=>`Error in gradient of depthwiseConv2d: number of input channels (${c.shape[3]}) must match the inChannels dimension in filter ${l.shape[2]}.`),b(ms(i,s),()=>`Error in gradient of depthwiseConv2d: Either strides or dilations must be  1. Got strides ${i} and dilations '${s}'.`),_s(`depthwiseConv2d`,a,o),{x:()=>up(c.shape,e,l,i,a,s,o),filter:()=>cp(c,e,l.shape,i,a,s,o)}}},a_={kernelName:Tt,inputsToSave:[`x`,`filter`],gradFunc:(e,t,n)=>{let[r,i]=t,a={x:r,filter:i,dy:e},o={x:r,filter:i,dy:e};return{x:()=>M.runKernel(Et,a,n),filter:()=>M.runKernel(Dt,o,n)}}},o_={kernelName:`Elu`,outputsToSave:[!0],gradFunc:(e,t)=>{let[n]=t,r={dy:e,y:n};return{x:()=>M.runKernel(jt,r)}}},s_={kernelName:`Erf`,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t,r=z(El(fu(vl(n))),2/Math.sqrt(Math.PI));return{x:()=>z(e,r)}}},c_={kernelName:`Exp`,outputsToSave:[!0],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(e,n)}}},l_={kernelName:Nt,inputsToSave:[`input`],gradFunc:(e,t)=>{let[n]=t;return{input:()=>B(e,n.shape)}}},u_={kernelName:Pt,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(e,El(n))}}},d_={kernelName:Lt,gradFunc:e=>({x:()=>Wc(e)})},f_={kernelName:Rt,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t,i=V(n.shape,r.shape);return{a:()=>{let t=R(e,I(r,`float32`)),a=Rc(n.shape,i);return a.length>0?B(U(t,a),n.shape):t},b:()=>{let t=z(e,I(n,`float32`)),a=Rc(r.shape,i);a.length>0&&(t=B(U(t,a),r.shape));let o=vl(r);return fu(R(t,I(o,`float32`)))}}}},p_={kernelName:zt,inputsToSave:[`x`,`mean`,`variance`,`scale`],gradFunc:(e,t,n)=>{let{varianceEpsilon:r}=n,[i,a,o,s]=t,c=s??H(1),l=Rc(a.shape,i.shape),u=[];if(a.rank===1){for(let e=0;e<i.shape.length-1;++e)u.push(i.shape[e]);u.push(1)}let d=W(i,a),f=z(e,c),p=Bd(L(o,H(r))),m=z(z(z(p,p),p),H(-.5));return{x:()=>a.rank===1?B(z(z(e,Ml(B(p,[1,1,1,a.shape[0]]),u)),c),i.shape):B(z(z(e,p),c),i.shape),mean:()=>{let e=z(z(p,H(-1)),f);return a.rank===1&&(e=U(e,l)),B(e,a.shape)},variance:()=>{let e=z(z(m,d),f);return a.rank===1&&(e=U(e,l)),B(e,a.shape)},scale:()=>{let t=z(e,z(d,p));return a.rank===1&&(t=U(t,l)),B(t,a.shape)},offset:()=>{let t=e;return a.rank===1&&(t=U(t,l)),B(t,a.shape)}}}},m_={kernelName:Bt,inputsToSave:[`x`,`indices`],gradFunc:(e,t,n)=>{let[r,i]=t,{axis:a,batchDims:o}=n,s=O(a,r.shape)[0],c=(e,t,n)=>()=>{let r=e.shape,i=t.size,o=r.slice(0,s),c=o.length,l=r.slice(a,r.length).slice(1),u=l.length,d=h_(0,c),f=h_(c+1,c+1+u),p=B(n,g_([o,[i],l])),m=B(t,[i]),h=g_([[c],d,f]),g=Uf(Yf(p,h),m,e.shape[s]),_=sl(h);return g=Yf(g,_),g};if(o===1){let t=r.shape[0],n=r.split(t,0);return{x:()=>Cf(n.map((t,n)=>c(t,i.slice(n,1),e.slice(n,1))())).reshape(r.shape),indices:()=>i}}return{x:c(r,i,e),indices:()=>i}}};function h_(e,t){let n=[];for(let r=e;r<t;++r)n.push(r);return n}function g_(e){let t=[];for(let n=0;n<e.length;++n)for(let r=0;r<e[n].length;++r)t.push(e[n][r]);return t}var __={kernelName:Ut,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t;return{a:()=>Wc(n),b:()=>Wc(r)}}},v_={kernelName:Wt,gradFunc:e=>({x:()=>I(e,`float32`)})},y_={kernelName:qt,gradFunc:e=>({x:()=>Wc(e)})},b_={kernelName:Jt,gradFunc:e=>({x:()=>Wc(e)})},x_={kernelName:Yt,gradFunc:e=>({x:()=>Wc(e)})},S_={kernelName:Xt,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{alpha:i}=n,a=Bl(r,0);return{x:()=>Hc(a,e,z(e,i))}}},C_={kernelName:en,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,L(n,1))}}},w_={kernelName:`Log`,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,I(n,`float32`))}}},T_={kernelName:an,inputsToSave:[],outputsToSave:[!0],gradFunc:(e,t,n)=>{let[r]=t,{axis:i}=n;return{logits:()=>{let t=El(r);return W(e,z(U(e,i,!0),t))}}}};function E_(e,t,n,r=5,i=1,a=1,o=.5){let s={x:e,y:t,dy:n},c={depthRadius:r,bias:i,alpha:a,beta:o};return M.runKernel(on,s,c)}var D_=P({localResponseNormalizationBackprop_:E_}),O_={kernelName:`LRN`,inputsToSave:[`x`],outputsToSave:[!0],gradFunc:(e,t,n)=>{let[r,i]=t,{depthRadius:a,bias:o,alpha:s,beta:c}=n;return{x:()=>D_(r,i,e,a,o,s,c)}}};function k_(e,t,n,r){return t.rank<n.rank&&(t=B(t,il(t.shape,r))),e.rank<n.rank&&(e=B(e,il(e.shape,r))),{x:()=>z(e,I(Bc(n,t),e.dtype))}}var A_={kernelName:`Max`,inputsToSave:[`x`],outputsToSave:[!0],gradFunc:(e,t,n)=>{let{reductionIndices:r}=n,i=t[0],a=t[1],o=k_(e,a,i,O(r,i.shape));return{x:()=>o.x()}}},j_={kernelName:sn,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t;return{a:()=>z(e,I(Hl(n,r),`float32`)),b:()=>z(e,I(eu(n,r),`float32`))}}};function M_(e,t,n,r,i,a,o){let s=N(e,`dy`,`maxPool3dGrad`),c=N(t,`input`,`maxPool3dGrad`),l=N(n,`output`,`maxPool3dGrad`),u=s,d=c,f=l,p=!1;c.rank===4&&(p=!0,u=B(s,[1,s.shape[0],s.shape[1],s.shape[2],s.shape[3]]),d=B(c,[1,c.shape[0],c.shape[1],c.shape[2],c.shape[3]]),f=B(l,[1,l.shape[0],l.shape[1],l.shape[2],l.shape[3]])),b(u.rank===5,()=>`Error in maxPool3dGrad: dy must be rank 5 but got rank ${u.rank}.`),b(d.rank===5,()=>`Error in maxPool3dGrad: input must be rank 5 but got rank ${d.rank}.`),b(f.rank===5,()=>`Error in maxPool3dGrad: output must be rank 5 but got rank ${f.rank}.`),_s(`maxPool3dGrad`,a,o);let m={dy:u,input:d,output:f},h={filterSize:r,strides:i,pad:a,dimRoundingMode:o},g=M.runKernel(dn,m,h);return p?B(g,[g.shape[1],g.shape[2],g.shape[3],g.shape[4]]):g}var N_=P({maxPool3dGrad_:M_}),P_={kernelName:un,inputsToSave:[`x`],outputsToSave:[!0],gradFunc:(e,t,n)=>{let[r,i]=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=n;return{x:()=>N_(e,r,i,a,o,s,c)}}};function F_(e,t,n,r,i,a,o){let s=N(e,`dy`,`maxPoolGrad`),c=N(t,`input`,`maxPoolGrad`),l=N(n,`output`,`maxPoolGrad`);b(c.rank===s.rank,()=>`Rank of input (${c.rank}) does not match rank of dy (${s.rank})`),b(s.rank===4,()=>`Error in maxPoolGrad: dy must be rank 4 but got rank ${s.rank}.`),b(c.rank===4,()=>`Error in maxPoolGrad: input must be rank 4 but got rank ${c.rank}.`),_s(`maxPoolGrad`,a,o);let u={dy:s,input:c,output:l},d={filterSize:r,strides:i,pad:a,dimRoundingMode:o};return M.runKernel(ln,u,d)}var I_=P({maxPoolGrad_:F_}),L_={kernelName:cn,inputsToSave:[`x`],outputsToSave:[!0],gradFunc:(e,t,n)=>{let[r,i]=t,{filterSize:a,strides:o,pad:s}=n;return{x:()=>I_(e,r,i,a,o,s)}}},R_={kernelName:pn,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{axis:i}=n,a=O(i,r.shape),o=rl(r.shape,a)[1],s=C(o);return{x:()=>{let t=r.shape.slice();return a.forEach(e=>{t[e]=1}),R(z(B(e,t),zu(r.shape,`float32`)),s)}}}},z_={kernelName:`Min`,inputsToSave:[`x`],outputsToSave:[!0],gradFunc:(e,t,n)=>{let{axis:r}=n,[i,a]=t,o=k_(e,a,i,O(r,i.shape));return{x:()=>o.x()}}},B_={kernelName:mn,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t;return{a:()=>z(e,I(nu(n,r),`float32`)),b:()=>z(e,I(Bl(n,r),`float32`))}}},V_={kernelName:hn,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let r=t[0],{paddings:i}=n,a=i.map(e=>e[0]);return{x:()=>As(e,a,r.shape)}}},H_={kernelName:`Mod`,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t,i=V(n.shape,r.shape);return{a:()=>{let t=Rc(n.shape,i);return t.length>0?B(U(e,t),n.shape):e},b:()=>{let t=z(e,fu(Il(R(n,r)))),a=Rc(r.shape,i);return a.length>0?B(U(t,a),r.shape):t}}}},U_={kernelName:_n,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t,i=V(n.shape,r.shape);return{a:()=>{let t=z(e,I(r,`float32`)),a=Rc(n.shape,i);return a.length>0?B(U(t,a),n.shape):t},b:()=>{let t=z(e,I(n,`float32`)),a=Rc(r.shape,i);return a.length>0?B(U(t,a),r.shape):t}}}},W_={kernelName:`Neg`,gradFunc:e=>({x:()=>fu(e)})},G_={kernelName:Cn,inputsToSave:[`indices`],gradFunc:(e,t)=>{let n=t[0];return{indices:()=>Ru(n.shape,`float32`)}}},K_={kernelName:Sn,gradFunc:e=>({x:()=>Wc(e)})},q_={kernelName:wn,saveAllInputs:!0,gradFunc:(e,t,n)=>{let{axis:r}=n;return Gf(e,r).map(e=>()=>e)}},J_={kernelName:Tn,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let r=t[0],{paddings:i}=n,a=i.map(e=>e[0]);return{x:()=>As(e,a,r.shape)}}},Y_={kernelName:`Pow`,inputsToSave:[`a`,`b`],outputsToSave:[!0],gradFunc:(e,t)=>{let[n,r,i]=t,a=n,o=r,s=V(a.shape,o.shape);return{a:()=>{let t=I(o,`float32`),n=z(e,z(t,ml(a,W(t,H(1))))),r=Rc(a.shape,s);return r.length>0&&(n=U(n,r)),B(n,a.shape)},b:()=>{let t=Hc(Bl(a,0),ou(a),Wc(a)),n=z(e,z(i,t)),r=Rc(o.shape,s);return r.length>0&&(n=U(n,r)),B(n,o.shape)}}}},X_={kernelName:En,inputsToSave:[`x`,`alpha`],gradFunc:(e,t)=>{let[n,r]=t,i=Bl(n,0);return{x:()=>Hc(i,e,z(e,r)),alpha:()=>{let t=Hc(i,Wc(e),z(e,n)),a=Rc(r.shape,e.shape);return a.length>0&&(t=U(t,a)),B(t,r.shape)}}}};function Z_(e,t,n){let r=e.shape.slice();return r[n]=1,z(B(t,r),z(Ec(e,n,!0,!1),Ec(e,n,!0,!0)))}function Q_(e,t,n){let r=e.shape.length,i=r-n.length,a=ol(n,r),o=e;a!=null&&(o=Yf(e,a));let s=o.shape.slice(),c=s.splice(r-n.length,n.length).reduce((e,t)=>e*t,1);s.push(c);let l=Z_(o.reshape(s),t,i);if(l=l.reshape(o.shape),a!=null){let e=sl(a);l=Yf(l,e)}return l}var $_={kernelName:Dn,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{axis:i}=n,a=[];return a=i==null?r.shape.map((e,t)=>t):typeof i==`number`?[i]:i,{x:()=>Q_(r,e,a)}}},ev={kernelName:kt,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t,i=V(n.shape,r.shape);return{a:()=>{let t=R(e,I(r,`float32`)),a=Rc(n.shape,i);return a.length>0?B(U(t,a),n.shape):t},b:()=>{let t=z(e,I(n,`float32`)),a=Rc(r.shape,i);a.length>0&&(t=B(U(t,a),r.shape));let o=vl(r);return fu(R(t,I(o,`float32`)))}}}},tv={kernelName:Nn,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,fu(vl(n)))}}},nv={kernelName:Bn,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t,r=z(nu(n,6),Tf(n));return{x:()=>z(e,I(r,`float32`))}}},rv={kernelName:Pn,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(e,I(Tf(n),`float32`))}}},iv={kernelName:Fn,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>B(e,n.shape)}}},av={kernelName:Rn,inputsToSave:[`images`],gradFunc:(e,t,n)=>{let[r]=t,i={dy:e,images:r};return{images:()=>M.runKernel(zn,i,n)}}},ov={kernelName:In,inputsToSave:[`images`],gradFunc:(e,t,n)=>{let[r]=t,i={dy:e,images:r};return{images:()=>M.runKernel(Ln,i,n)}}},sv={kernelName:Vn,gradFunc:(e,t,n)=>{let{dims:r}=n,i=O(r,e.shape);return{x:()=>Id(e,i)}}},cv={kernelName:Hn,gradFunc:e=>({x:()=>Wc(e)})},lv={kernelName:Un,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>fu(R(e,z(ml(n,1.5),2)))}}},uv={kernelName:qn,inputsToSave:[`condition`],gradFunc:(e,t)=>{let[n]=t;return{condition:()=>I(Wc(n),`float32`),t:()=>z(e,I(n,e.dtype)),e:()=>z(e,I(Tu(n),e.dtype))}}},dv={kernelName:Jn,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>{let t=Bl(n,H(0)),r=H(Mh);return Hc(t,z(e,H(Nh)),z(z(e,r),El(I(n,`float32`))))}}}},fv={kernelName:Qn,outputsToSave:[!0],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(e,z(n,W(H(1),n)))}}},pv={kernelName:Zn,gradFunc:e=>({x:()=>Wc(e)})},mv={kernelName:`Sin`,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(Sc(I(n,`float32`)),e)}}},hv={kernelName:Xn,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(wc(I(n,`float32`)),e)}}},gv={kernelName:Yn,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{begin:i,size:a}=n,o=r.shape,[s,c]=dh(r,i,a),l=[];for(let t=0;t<e.rank;t++)l.push([s[t],o[t]-s[t]-c[t]]);return{x:()=>td(e,l)}}},_v={kernelName:rr,outputsToSave:[!0],gradFunc:(e,t,n)=>{let[r]=t,{dim:i}=n,a=z(e,r);return{logits:()=>W(a,z(U(a,[i],!0),r))}}},vv={kernelName:$n,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(e,Os(n))}}},yv={kernelName:tr,gradFunc:(e,t,n)=>{let{blockShape:r,paddings:i}=n;return{x:()=>Ps(e,r,i)}}},bv={kernelName:nr,gradFunc:(e,t,n)=>{let{axis:r}=n;return{x:()=>ws(e,r)}}},xv={kernelName:er,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,z(gl(I(n,`float32`)),2))}}},Sv={kernelName:ur,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(e,z(I(n,`float32`),2))}}},Cv={kernelName:lr,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t,i=H(2);return{a:()=>z(e,z(i,W(n,r))),b:()=>z(e,z(i,W(r,n)))}}},wv={kernelName:Tr,gradFunc:e=>({x:()=>Wc(e)})},Tv={kernelName:`Sub`,inputsToSave:[`a`,`b`],gradFunc:(e,t)=>{let[n,r]=t,i=V(n.shape,r.shape);return{a:()=>{let t=e,r=Rc(n.shape,i);return r.length>0&&(t=U(t,r)),B(t,n.shape)},b:()=>{let t=e,n=Rc(r.shape,i);return n.length>0&&(t=U(t,n)),B(fu(t),r.shape)}}}},Ev={kernelName:`Sum`,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,i=r.shape.slice(),{axis:a}=n;O(a,r.shape).forEach(e=>{i[e]=1});let o=z(B(e,i),zu(r.shape,`float32`));return{x:()=>o}}},Dv={kernelName:`Tan`,inputsToSave:[`x`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>R(e,vl(Sc(n)))}}},Ov={kernelName:gr,outputsToSave:[!0],gradFunc:(e,t)=>{let[n]=t;return{x:()=>z(W(H(1),vl(n)),e)}}},kv={kernelName:_r,inputsToSave:[`x`],gradFunc:(e,t,n)=>{let[r]=t,{reps:i}=n;return{x:()=>{let t=Wc(r);if(r.rank===1)for(let n=0;n<i[0];++n)t=L(t,As(e,[n*r.shape[0]],[r.shape[0]]));else if(r.rank===2)for(let n=0;n<i[0];++n)for(let a=0;a<i[1];++a)t=L(t,As(e,[n*r.shape[0],a*r.shape[1]],[r.shape[0],r.shape[1]]));else if(r.rank===3)for(let n=0;n<i[0];++n)for(let a=0;a<i[1];++a)for(let o=0;o<i[2];++o)t=L(t,As(e,[n*r.shape[0],a*r.shape[1],o*r.shape[2]],[r.shape[0],r.shape[1],r.shape[2]]));else if(r.rank===4)for(let n=0;n<i[0];++n)for(let a=0;a<i[1];++a)for(let o=0;o<i[2];++o)for(let s=0;s<i[3];++s)t=L(t,As(e,[n*r.shape[0],a*r.shape[1],o*r.shape[2],s*r.shape[3]],[r.shape[0],r.shape[1],r.shape[2],r.shape[3]]));else throw Error(`Gradient for tile operation is not implemented for rank-${r.rank} tensors yet.`);return t}}}},Av={kernelName:br,gradFunc:(e,t,n)=>{let{perm:r}=n,i=sl(r);return{x:()=>Yf(e,i)}}},jv={kernelName:Sr,gradFunc:(e,t,n)=>{let{axis:r}=n;return{value:()=>Cf(e,r)}}},Mv={kernelName:Cr,inputsToSave:[`segmentIds`],gradFunc:(e,t)=>{let[n]=t;return{x:()=>Nv(e,n)}}};function Nv(e,t){let n=Rl(e,Fu(t,Wc(t))),r=Hl(t,H(0,`int32`)),i=n.rank-r.rank;for(let e=0;e<i;++e)r=Ol(r,e+1);r=Cu(r,zu(n.shape,`bool`));let a=Wc(n);return Hc(r,n,a)}var Pv=[wg,Tg,Eg,Dg,Og,kg,Ag,jg,Mg,Ng,Pg,Fg,Rg,Vg,Hg,Ug,Wg,Gg,Kg,qg,Jg,Yg,Zg,Xg,e_,t_,n_,r_,i_,a_,ev,o_,s_,c_,l_,u_,f_,d_,p_,m_,__,v_,y_,b_,x_,S_,C_,w_,T_,O_,A_,A_,j_,P_,L_,R_,z_,B_,V_,H_,U_,W_,G_,K_,q_,J_,J_,Y_,X_,$_,tv,nv,rv,iv,av,ov,sv,cv,lv,uv,dv,fv,pv,mv,hv,gv,_v,vv,yv,yv,bv,bv,xv,Cv,Sv,wv,Tv,Ev,Dv,Ov,kv,Av,jv,Mv,{kernelName:wr,gradFunc:e=>({x:()=>Wc(e)})}];for(let e of Pv)zr(e);j().prototype.abs=function(){return this.throwIfDisposed(),Oo(this)},j().prototype.acos=function(){return this.throwIfDisposed(),Ao(this)},j().prototype.acosh=function(){return this.throwIfDisposed(),Mo(this)},j().prototype.add=function(e){return this.throwIfDisposed(),L(this,e)},j().prototype.all=function(e,t){return this.throwIfDisposed(),Io(this,e,t)},j().prototype.any=function(e,t){return this.throwIfDisposed(),Ro(this,e,t)},j().prototype.argMax=function(e){return this.throwIfDisposed(),Bo(this,e)},j().prototype.argMin=function(e){return this.throwIfDisposed(),Ho(this,e)},j().prototype.asScalar=function(){return this.throwIfDisposed(),b(this.size===1,()=>`The array must have only 1 element.`),B(this,[])},j().prototype.asType=function(e){return this.throwIfDisposed(),I(this,e)},j().prototype.as1D=function(){return this.throwIfDisposed(),B(this,[this.size])},j().prototype.as2D=function(e,t){return this.throwIfDisposed(),B(this,[e,t])},j().prototype.as3D=function(e,t,n){return this.throwIfDisposed(),B(this,[e,t,n])},j().prototype.as4D=function(e,t,n,r){return this.throwIfDisposed(),B(this,[e,t,n,r])},j().prototype.as5D=function(e,t,n,r,i){return this.throwIfDisposed(),B(this,[e,t,n,r,i])},j().prototype.asin=function(){return this.throwIfDisposed(),Wo(this)},j().prototype.asinh=function(){return this.throwIfDisposed(),Ko(this)},j().prototype.atan=function(){return this.throwIfDisposed(),Jo(this)},j().prototype.atan2=function(e){return this.throwIfDisposed(),Xo(this,e)},j().prototype.atanh=function(){return this.throwIfDisposed(),Qo(this)},j().prototype.avgPool=function(e,t,n,r){return this.throwIfDisposed(),bs(this,e,t,n,r)},j().prototype.batchToSpaceND=function(e,t){return this.throwIfDisposed(),Ps(this,e,t)},j().prototype.batchNorm=function(e,t,n,r,i){return this.throwIfDisposed(),Ls(this,e,t,n,r,i)},j().prototype.broadcastTo=function(e){return this.throwIfDisposed(),qs(this,e)},j().prototype.cast=function(e){return this.throwIfDisposed(),I(this,e)},j().prototype.ceil=function(){return this.throwIfDisposed(),Ys(this)},j().prototype.clipByValue=function(e,t){return this.throwIfDisposed(),Qs(this,e,t)},j().prototype.concat=function(e,t){return this.throwIfDisposed(),e instanceof Ni&&(e=[e]),ws([this,...e],t)},j().prototype.conv1d=function(e,t,n,r,i,a){return this.throwIfDisposed(),uc(this,e,t,n,r,i,a)},j().prototype.conv2dTranspose=function(e,t,n,r,i){return this.throwIfDisposed(),mc(this,e,t,n,r,i)},j().prototype.conv2d=function(e,t,n,r,i,a){return this.throwIfDisposed(),cc(this,e,t,n,r,i,a)},j().prototype.cos=function(){return this.throwIfDisposed(),Sc(this)},j().prototype.cosh=function(){return this.throwIfDisposed(),wc(this)},j().prototype.cumprod=function(e,t,n){return this.throwIfDisposed(),Ec(this,e,t,n)},j().prototype.cumsum=function(e,t,n){return this.throwIfDisposed(),Oc(this,e,t,n)},j().prototype.depthToSpace=function(e,t){return this.throwIfDisposed(),Mc(this,e,t)},j().prototype.depthwiseConv2d=function(e,t,n,r,i,a){return this.throwIfDisposed(),Pc(this,e,t,n,r,i,a)},j().prototype.dilation2d=function(e,t,n,r,i){return this.throwIfDisposed(),Ic(this,e,t,n,r,i)},j().prototype.divNoNan=function(e){return this.throwIfDisposed(),Kc(this,e)},j().prototype.div=function(e){return this.throwIfDisposed(),R(this,e)},j().prototype.dot=function(e){return this.throwIfDisposed(),Jc(this,e)},j().prototype.elu=function(){return this.throwIfDisposed(),Qc(this)},j().prototype.equal=function(e){return this.throwIfDisposed(),Bc(this,e)},j().prototype.erf=function(){return this.throwIfDisposed(),el(this)},j().prototype.euclideanNorm=function(e,t){return this.throwIfDisposed(),wl(this,e,t)},j().prototype.exp=function(){return this.throwIfDisposed(),El(this)},j().prototype.expandDims=function(e){return this.throwIfDisposed(),Ol(this,e)},j().prototype.expm1=function(){return this.throwIfDisposed(),Al(this)},j().prototype.fft=function(){return this.throwIfDisposed(),lf(this)},j().prototype.flatten=function(){return this.throwIfDisposed(),B(this,[this.size])},j().prototype.floor=function(){return this.throwIfDisposed(),Il(this)},j().prototype.floorDiv=function(e){return this.throwIfDisposed(),wo(this,e)},j().prototype.gather=function(e,t,n){return this.throwIfDisposed(),Rl(this,e,t,n)},j().prototype.greaterEqual=function(e){return this.throwIfDisposed(),Hl(this,e)},j().prototype.greater=function(e){return this.throwIfDisposed(),Bl(this,e)},j().prototype.ifft=function(){return this.throwIfDisposed(),df(this)},j().prototype.irfft=function(){return this.throwIfDisposed(),pf(this)},j().prototype.isFinite=function(){return this.throwIfDisposed(),Kl(this)},j().prototype.isInf=function(){return this.throwIfDisposed(),Jl(this)},j().prototype.isNaN=function(){return this.throwIfDisposed(),Xl(this)},j().prototype.leakyRelu=function(e){return this.throwIfDisposed(),Ql(this,e)},j().prototype.lessEqual=function(e){return this.throwIfDisposed(),nu(this,e)},j().prototype.less=function(e){return this.throwIfDisposed(),eu(this,e)},j().prototype.localResponseNormalization=function(e,t,n,r){return this.throwIfDisposed(),iu(this,e,t,n,r)},j().prototype.logSigmoid=function(){return this.throwIfDisposed(),gu(this)},j().prototype.logSoftmax=function(e){return this.throwIfDisposed(),yu(this,e)},j().prototype.logSumExp=function(e,t){return this.throwIfDisposed(),xu(this,e,t)},j().prototype.log=function(){return this.throwIfDisposed(),ou(this)},j().prototype.log1p=function(){return this.throwIfDisposed(),cu(this)},j().prototype.logicalAnd=function(e){return this.throwIfDisposed(),Cu(this,e)},j().prototype.logicalNot=function(){return this.throwIfDisposed(),Tu(this)},j().prototype.logicalOr=function(e){return this.throwIfDisposed(),Du(this,e)},j().prototype.logicalXor=function(e){return this.throwIfDisposed(),ku(this,e)},j().prototype.matMul=function(e,t,n){return this.throwIfDisposed(),Es(this,e,t,n)},j().prototype.maxPool=function(e,t,n,r){return this.throwIfDisposed(),ju(this,e,t,n,r)},j().prototype.max=function(e,t){return this.throwIfDisposed(),ul(this,e,t)},j().prototype.maximum=function(e){return this.throwIfDisposed(),Fu(this,e)},j().prototype.mean=function(e,t){return this.throwIfDisposed(),Lu(this,e,t)},j().prototype.min=function(e,t){return this.throwIfDisposed(),fl(this,e,t)},j().prototype.minimum=function(e){return this.throwIfDisposed(),Vu(this,e)},j().prototype.mirrorPad=function(e,t){return this.throwIfDisposed(),Uu(this,e,t)},j().prototype.mod=function(e){return this.throwIfDisposed(),Gu(this,e)},j().prototype.mul=function(e){return this.throwIfDisposed(),z(this,e)},j().prototype.neg=function(){return this.throwIfDisposed(),fu(this)},j().prototype.norm=function(e,t,n){return this.throwIfDisposed(),Sl(this,e,t,n)},j().prototype.notEqual=function(e){return this.throwIfDisposed(),Yu(this,e)},j().prototype.oneHot=function(e,t=1,n=0){return this.throwIfDisposed(),Zu(this,e,t,n)},j().prototype.onesLike=function(){return this.throwIfDisposed(),$u(this)},j().prototype.pad=function(e,t){return this.throwIfDisposed(),td(this,e,t)},j().prototype.pool=function(e,t,n,r,i,a){return this.throwIfDisposed(),sd(this,e,t,n,r,i,a)},j().prototype.pow=function(e){return this.throwIfDisposed(),ml(this,e)},j().prototype.prelu=function(e){return this.throwIfDisposed(),ld(this,e)},j().prototype.prod=function(e,t){return this.throwIfDisposed(),dd(this,e,t)},j().prototype.reciprocal=function(){return this.throwIfDisposed(),Ad(this)},j().prototype.relu=function(){return this.throwIfDisposed(),Md(this)},j().prototype.relu6=function(){return this.throwIfDisposed(),Pd(this)},j().prototype.reshapeAs=function(e){return this.throwIfDisposed(),B(this,e.shape)},j().prototype.reshape=function(e){return this.throwIfDisposed(),B(this,e)},j().prototype.resizeBilinear=function(e,t,n){return this.throwIfDisposed(),Jp(this,e,t,n)},j().prototype.resizeNearestNeighbor=function(e,t,n){return this.throwIfDisposed(),Xp(this,e,t,n)},j().prototype.reverse=function(e){return this.throwIfDisposed(),Id(this,e)},j().prototype.rfft=function(){return this.throwIfDisposed(),_f(this)},j().prototype.round=function(){return this.throwIfDisposed(),Rd(this)},j().prototype.rsqrt=function(){return this.throwIfDisposed(),Bd(this)},j().prototype.selu=function(){return this.throwIfDisposed(),Hd(this)},j().prototype.separableConv2d=function(e,t,n,r,i,a){return this.throwIfDisposed(),Wd(this,e,t,n,r,i,a)},j().prototype.sigmoid=function(){return this.throwIfDisposed(),Os(this)},j().prototype.sign=function(){return this.throwIfDisposed(),Kd(this)},j().prototype.sin=function(){return this.throwIfDisposed(),Jd(this)},j().prototype.sinh=function(){return this.throwIfDisposed(),Xd(this)},j().prototype.slice=function(e,t){return this.throwIfDisposed(),As(this,e,t)},j().prototype.softmax=function(e){return this.throwIfDisposed(),sf(this,e)},j().prototype.softplus=function(){return this.throwIfDisposed(),mu(this)},j().prototype.spaceToBatchND=function(e,t){return this.throwIfDisposed(),rd(this,e,t)},j().prototype.split=function(e,t){return this.throwIfDisposed(),hf(this,e,t)},j().prototype.sqrt=function(){return this.throwIfDisposed(),gl(this)},j().prototype.square=function(){return this.throwIfDisposed(),vl(this)},j().prototype.squaredDifference=function(e){return this.throwIfDisposed(),yf(this,e)},j().prototype.squeeze=function(e){return this.throwIfDisposed(),xf(this,e)},j().prototype.stack=function(e,t){return this.throwIfDisposed(),Cf(e instanceof Ni?[this,e]:[this,...e],t)},j().prototype.step=function(e){return this.throwIfDisposed(),Tf(this,e)},j().prototype.stridedSlice=function(e,t,n,r,i,a,o,s){return this.throwIfDisposed(),Df(this,e,t,n,r,i,a,o,s)},j().prototype.sub=function(e){return this.throwIfDisposed(),W(this,e)},j().prototype.sum=function(e,t){return this.throwIfDisposed(),U(this,e,t)},j().prototype.tan=function(){return this.throwIfDisposed(),kf(this)},j().prototype.tanh=function(){return this.throwIfDisposed(),Ms(this)},j().prototype.tile=function(e){return this.throwIfDisposed(),Ml(this,e)},j().prototype.toBool=function(){return this.throwIfDisposed(),I(this,`bool`)},j().prototype.toFloat=function(){return this.throwIfDisposed(),I(this,`float32`)},j().prototype.toInt=function(){return this.throwIfDisposed(),I(this,`int32`)},j().prototype.topk=function(e,t){return this.throwIfDisposed(),Lf(this,e,t)},j().prototype.transpose=function(e){return this.throwIfDisposed(),Yf(this,e)},j().prototype.unique=function(e){return this.throwIfDisposed(),Vf(this,e)},j().prototype.unsortedSegmentSum=function(e,t){return this.throwIfDisposed(),Uf(this,e,t)},j().prototype.unstack=function(e){return this.throwIfDisposed(),Gf(this,e)},j().prototype.where=function(e,t){return this.throwIfDisposed(),Hc(e,this,t)},j().prototype.zerosLike=function(){return this.throwIfDisposed(),Wc(this)};var Fv=class e extends Error{constructor(t){super(t),Object.setPrototypeOf(this,e.prototype)}},Iv=class e extends Error{constructor(t){super(t),Object.setPrototypeOf(this,e.prototype)}},K=class e extends Error{constructor(t){super(t),Object.setPrototypeOf(this,e.prototype)}},Lv=class e extends Error{constructor(t){super(t),Object.setPrototypeOf(this,e.prototype)}},Rv=class e extends Error{constructor(t){super(t),Object.setPrototypeOf(this,e.prototype)}},zv=class{constructor(e){this.maxEntries=e||100,this.cache=new Map}get(e){let t;return this.cache.has(e)&&(t=this.cache.get(e),this.cache.delete(e),this.cache.set(e,t)),t}put(e,t){if(this.cache.has(e))this.cache.delete(e);else if(this.cache.size>=this.maxEntries){let e=this.cache.keys().next().value;this.cache.delete(e)}this.cache.set(e,t)}getMaxEntries(){return this.maxEntries}setMaxEntries(e){if(e<0)throw Error(`The maxEntries of LRU caches must be at least 0, but got ${e}.`);if(this.maxEntries>e)for(let t=0;t<this.maxEntries-e;t++){let e=this.cache.keys().next().value;this.cache.delete(e)}this.maxEntries=e}};function Bv(e,t){if(Array.isArray(e)){let n=[];for(let r=0;r<t;r++)n=n.concat(e);return n}{let n=Array(t);return n.fill(e),n}}function Vv(e,t){if(!e)throw new Rv(t)}function Hv(e,t){let n=0;for(let r of e)r===t&&n++;return n}function Uv(e){return e.length===1?e[0]:e}function Wv(e){return Array.isArray(e)?e:[e]}function Gv(e){let t=e.replace(/(.)([A-Z][a-z0-9]+)/g,`$1_$2`).replace(/([a-z])([A-Z])/g,`$1_$2`).toLowerCase();return t[0]===`_`?`private`+t:t}function Kv(e){return e.length<=1||e.indexOf(`_`)===-1?e:e.replace(/[_]+(\w|$)/g,(e,t)=>t.toUpperCase())}var qv={};function Jv(e){if(e==null)return null;let t={};return t.className=e.getClassName(),t.config=e.getConfig(),t}function Yv(e){if(typeof e==`object`&&e){if(Array.isArray(e))e.forEach(e=>Yv(e));else{let t=Object.keys(e);for(let n of t){let t=e[n];typeof t==`object`&&t&&(!Array.isArray(t)&&t.type===`ndarray`&&typeof t.value==`number`?e[n]=t.value:Yv(t))}}}}function Xv(e,t={},n={},r=`object`,i=!1){if(typeof e==`string`){let i=e,a;if(i in n)a=n[i];else if(i in qv)a=qv[i];else if(a=t[i],a==null)throw new K(`Unknown ${r}: ${e}. This may be due to one of the following reasons:\n1. The ${r} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.\n2. The custom ${r} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);return a}{let a=e;if(a.className==null||a.config==null)throw new K(`${r}: Improper config format: ${JSON.stringify(a)}.\n'className' and 'config' must set.`);let o=a.className,s,c;if(o in n?[s,c]=n[o]:o in qv?[s,c]=qv.className:o in t&&([s,c]=t[o]),s==null)throw new K(`Unknown ${r}: ${o}. This may be due to one of the following reasons:\n1. The ${r} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.\n2. The custom ${r} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);if(c!=null){let e={};for(let t of Object.keys(qv))e[t]=qv[t];for(let t of Object.keys(n))e[t]=n[t];let t=a.config;t.customObjects=e;let r=Object.assign({},qv);for(let e of Object.keys(n))qv[e]=n[e];Yv(a.config);let o=c(s,a.config,n,i);return qv=Object.assign({},r),o}{let e=Object.assign({},qv);for(let e of Object.keys(n))qv[e]=n[e];let t=new s(a.config);return qv=Object.assign({},e),t}}}function Zv(e,t){return e<t?-1:+(e>t)}function Qv(e,t){return-1*Zv(e,t)}function $v(e){if(e==null)return e;let t=[];for(let n of e)t.indexOf(n)===-1&&t.push(n);return t}function ey(e){if(e==null)throw new K(`Invalid value in obj: ${JSON.stringify(e)}`);for(let t in e)if(e.hasOwnProperty(t))return!1;return!0}function ty(e,t,n){if(n!=null&&e.indexOf(n)<0)throw new K(`${n} is not a valid ${t}.  Valid values are ${e} or null/undefined.`)}function ny(e,t,n=0,r=1/0){return Vv(n>=0),Vv(r>=n),Array.isArray(e)&&e.length>=n&&e.length<=r&&e.every(e=>typeof e===t)}function ry(e,t){Array.isArray(e)?(b(e.length>0,()=>`${t} is unexpectedly an empty array.`),e.forEach((e,n)=>ry(e,`element ${n+1} of ${t}`))):b(Number.isInteger(e)&&e>0,()=>`Expected ${t} to be a positive integer, but got ${iy(e)}.`)}function iy(e){return e===null?`null`:Array.isArray(e)?`[`+e.map(e=>iy(e)).join(`,`)+`]`:typeof e==`string`?`"${e}"`:`${e}`}function ay(e,t,n){let r=n==null?li():n(),i;return(...a)=>{let o=n==null?li():n();return o-r<t?i:(r=o,i=e(...a),i)}}function oy(e){return e===`relu`?`relu`:e===`linear`?`linear`:e===`elu`?`elu`:null}var sy=0;function cy(){return sy++}var ly={};function uy(e=``){return e in ly||(ly[e]=0),ly[e]+=1,e+ly[e].toString()}var dy=[`channelsFirst`,`channelsLast`],fy=[`nearest`,`bilinear`],py=[`valid`,`same`,`causal`],my=[`max`,`avg`],hy=[`sum`,`mul`,`concat`,`ave`],gy=new Map;function _y(e){ty(dy,`DataFormat`,e)}function vy(e){ty(fy,`InterpolationFormat`,e)}function yy(e){ty(py,`PaddingMode`,e)}function by(e){ty(my,`PoolMode`,e)}var xy=[],Sy=`/`;function Cy(e,t){xy.push(e);try{let e=t();return xy.pop(),e}catch(e){throw xy.pop(),e}}function wy(){return xy.length===0?``:xy.join(Sy)+Sy}function Ty(e){if(!Oy(e))throw Error(`Not a valid tensor name: '`+e+`'`);return wy()+e}function Ey(e){if(!Oy(e))throw Error(`Not a valid tensor name: '`+e+`'`);gy.has(e)||gy.set(e,0);let t=gy.get(e);if(gy.set(e,gy.get(e)+1),t>0){let n=`${e}_${t}`;return gy.set(n,1),n}return e}var Dy=new RegExp(/^[A-Za-z0-9][-A-Za-z0-9\._\/]*$/);function Oy(e){return!!e.match(Dy)}function ky(e){return e===parseInt(e.toString(),10)}function Ay(e,t,n){t??=0,n??=e.length;let r=1;for(let i=t;i<n;++i)r*=e[i];return r}function jy(e){if(e.length===0)return NaN;let t=1/0;for(let n=0;n<e.length;n++){let r=e[n];r<t&&(t=r)}return t}function My(e){if(e.length===0)return NaN;let t=-1/0;for(let n=0;n<e.length;n++){let r=e[n];r>t&&(t=r)}return t}function Ny(e,t){if(t<e)throw new K(`end (${t}) < begin (${e}) is forbidden.`);let n=[];for(let r=e;r<t;++r)n.push(r);return n}var Py;function Fy(){return Py??=Ta().epsilon(),Py}function Iy(){return`channelsLast`}function Ly(e,t){return I(e,t)}function Ry(e,t=-1){let n=e.shape.slice();return t<0&&(t=n.length+t+1),n.splice(t,0,1),B(e,n)}function zy(e,t){return F(()=>{if(e.shape.length!==2)throw new K(`repeat() expects a rank-2 tensor, but received a rank-${e.shape.length} tensor.`);return qy(Ry(e,1),[1,t,1])})}function By(e){return B(e,[Ay(e.shape)])}function Vy(e){if(e.rank<=1)throw new K(`batchFlatten requires a minimum rank of 2. Got rank: ${e.rank}.`);return B(e,[e.shape[0],Ay(e.shape,1)])}function Hy(e,t,n){return F(()=>{switch(e.rank){case 1:return Qd(e,t,n);case 2:return ef(e,[t,0],[n,e.shape[1]]);case 3:return nf(e,[t,0,0],[n,e.shape[1],e.shape[2]]);case 4:return af(e,[t,0,0,0],[n,e.shape[1],e.shape[2],e.shape[3]]);case 5:return As(e,[t,0,0,0,0],[n,e.shape[1],e.shape[2],e.shape[3],e.shape[4]]);case 6:return As(e,[t,0,0,0,0,0],[n,e.shape[1],e.shape[2],e.shape[3],e.shape[4],e.shape[5]]);default:throw new K(`sliceAlongFirstAxis() received an unsupported tensor rank: ${e.rank}`)}})}function Uy(e,t,n){return F(()=>{switch(e.rank){case 1:return Qd(e,t,n);case 2:return ef(e,[0,t],[e.shape[0],n]);case 3:return nf(e,[0,0,t],[e.shape[0],e.shape[1],n]);case 4:return af(e,[0,0,0,t],[e.shape[0],e.shape[1],e.shape[2],n]);default:throw new K(`sliceAlongLastAxis() received an unsupported tensor rank: ${e.rank}`)}})}function Wy(e,t,n,r){return F(()=>{switch(e.rank){case 1:return Qd(e,t,n);case 2:switch(r){case 1:return Hy(e,t,n);case 2:return Uy(e,t,n);default:throw new K(`The axis is not within the rank of the tensor ${r}`)}case 3:switch(r){case 1:return Hy(e,t,n);case 2:return nf(e,[0,t,0],[e.shape[0],n,e.shape[2]]);case 3:return Uy(e,t,n);default:throw new K(`The axis is not within the rank of the tensor ${r}`)}case 4:switch(r){case 1:return Hy(e,t,n);case 2:return af(e,[0,t,0,0],[e.shape[0],n,e.shape[2],e.shape[3]]);case 3:return af(e,[0,0,t,0],[e.shape[0],e.shape[1],n,e.shape[3]]);case 4:return Uy(e,t,n);default:throw new K(`The axis is not within the rank of the tensor ${r}`)}default:throw new K(`sliceAlongLastAxis() received an unsupported tensor rank: ${e.rank}`)}})}function Gy(e,t=-1){let n;return t<0&&(n=e[0].rank,t=n===0?0:n),t===e[0].rank&&(t=-1),ws(e,t)}function Ky(e,t){switch(e.rank){case 1:return ec([e,t]);case 2:return nc([e,t],0);case 3:return ic([e,t],0);case 4:return oc([e,t],0);default:throw new K(`concatAlongFirstAxis() received an unsupported tensor rank: ${e.rank}`)}}function qy(e,t){if(Array.isArray(t)||(t=[t]),e.rank!==t.length)throw new K(`The length of input n (${t.length}) does not match the number of dimensions in input x (${e.rank})`);return Ml(e,t)}function Jy(e,t=0,n=1,r,i){return Cd(e,t,n,r,i)}function Yy(e,t,n,r){if(e.rank<2||t.rank<2)throw new Lv(`dot requires both inputs to be rank >= 2 but got x shape = ${e.shape} and y shape = ${t.shape}`);if(t.rank>=3&&e.shape.slice(-1)[0]!==t.shape.slice(-2)[0])throw new Lv(`If rank y >= 3, then the second last dim of y must equal the last dim of x but got x shape = ${e.shape} and  y shape = ${t.shape}`);if(e.rank===2&&t.rank===2)return fp({a:e,b:t,transposeA:!1,transposeB:!1,bias:r?Qy(e.rank,r,Iy()):null,activation:n});{let i=e.shape.slice(),a=i.pop();e=B(e,[-1,a]);let o=t.shape.slice(),s=o.pop(),c=o.pop(),l=[...o,s],u=Array.from({length:t.rank},(e,n)=>n===0?t.rank-2:n<=t.rank-2?n-1:n);t=B(Yf(t,u),[c,-1]);let d=[...i,...l];return B(fp({a:e,b:t,transposeA:!1,transposeB:!1,bias:r?Qy(e.rank,r,Iy()):null,activation:n}),d)}}function Xy(e,t,n){return F(()=>(t=Array.isArray(t)?Af(t,`int32`):I(t,`int32`),Rl(e,t,n)))}function Zy(e){return z(e,e)}function Qy(e,t,n){let r=t.shape;if(t.rank!==1&&t.rank!==e)throw new K(`Unexpected bias dimensions: ${t.rank}; expected it to be 1 or ${e}`);if(e===5){if(n===`channelsFirst`)return r.length===1?B(t,[1,r[0],1,1,1]):B(t,[1,r[3],r[0],r[1],r[2]]);if(n===`channelsLast`)return r.length===1?B(t,[1,1,1,1,r[0]]):B(t,[1].concat(r))}else if(e===4){if(n===`channelsFirst`)return r.length===1?B(t,[1,r[0],1,1]):B(t,[1,r[2],r[0],r[1]]);if(n===`channelsLast`)return r.length===1?B(t,[1,1,1,r[0]]):B(t,[1].concat(r))}else if(e===3){if(n===`channelsFirst`)return r.length===1?B(t,[1,r[0],1]):B(t,[1,r[1],r[0]]);if(n===`channelsLast`)return r.length===1?B(t,[1,1,r[0]]):B(t,[1].concat(r))}else if(e<3)return t;throw new K(`Unsupported input rank by biasAdd: ${t.rank}`)}function $y(e,t,n){return F(()=>(n??=Iy(),_y(n),L(e,Qy(e.rank,t,n))))}function eb(e,t=1){if(t!==1)throw new Lv(`Support for alpha values other than 1 (${t}) is not implemented yet.`);return Qc(e)}function tb(e){return F(()=>R(e,L(Oo(e),1)))}function nb(e,t,n,r){return F(()=>Qf(e,t,n,r))}function rb(e){return F(()=>Qs(L(.5,z(.2,e)),0,1))}function ib(e,t,n=!1){return n?e():t()}var ab=[`fanIn`,`fanOut`,`fanAvg`],ob=[`normal`,`uniform`,`truncatedNormal`];function sb(e){ty(ab,`FanMode`,e)}function cb(e){ty(ob,`Distribution`,e)}var lb=class extends Fm{fromConfigUsesCustomObjects(){return!1}getConfig(){return{}}},ub=class extends lb{apply(e,t){return Ru(e,t)}};ub.className=`Zeros`,G(ub);var db=class extends lb{apply(e,t){return zu(e,t)}};db.className=`Ones`,G(db);var fb=class extends lb{constructor(e){if(super(),typeof e!=`object`)throw new K(`Expected argument of type ConstantConfig but got ${e}`);if(e.value===void 0)throw new K(`config must have value set but got ${e}`);this.value=e.value}apply(e,t){return F(()=>z(H(this.value),zu(e,t)))}getConfig(){return{value:this.value}}};fb.className=`Constant`,G(fb);var pb=class extends lb{constructor(e){super(),this.DEFAULT_MINVAL=-.05,this.DEFAULT_MAXVAL=.05,this.minval=e.minval||this.DEFAULT_MINVAL,this.maxval=e.maxval||this.DEFAULT_MAXVAL,this.seed=e.seed}apply(e,t){return Td(e,this.minval,this.maxval,t,this.seed)}getConfig(){return{minval:this.minval,maxval:this.maxval,seed:this.seed}}};pb.className=`RandomUniform`,G(pb);var mb=class extends lb{constructor(e){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=e.mean||this.DEFAULT_MEAN,this.stddev=e.stddev||this.DEFAULT_STDDEV,this.seed=e.seed}apply(e,t){if(t||=`float32`,t!==`float32`&&t!==`int32`)throw new Lv(`randomNormal does not support dType ${t}.`);return Jy(e,this.mean,this.stddev,t,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}};mb.className=`RandomNormal`,G(mb);var hb=class extends lb{constructor(e){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=e.mean||this.DEFAULT_MEAN,this.stddev=e.stddev||this.DEFAULT_STDDEV,this.seed=e.seed}apply(e,t){if(t||=`float32`,t!==`float32`&&t!==`int32`)throw new Lv(`truncatedNormal does not support dType ${t}.`);return zf(e,this.mean,this.stddev,t,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}};hb.className=`TruncatedNormal`,G(hb);var gb=class extends lb{constructor(e){super(),this.gain=e.gain==null?1:e.gain}apply(e,t){return F(()=>{if(e.length!==2||e[0]!==e[1])throw new K(`Identity matrix initializer can only be used for 2D square matrices.`);return z(this.gain,Pl(e[0]))})}getConfig(){return{gain:this.gain}}};gb.className=`Identity`,G(gb);function _b(e,t=`channelsLast`){let n,r;if(_y(t),e.length===2)n=e[0],r=e[1];else if([3,4,5].indexOf(e.length)!==-1){if(t===`channelsFirst`){let t=Ay(e,2);n=e[1]*t,r=e[0]*t}else if(t===`channelsLast`){let t=Ay(e,0,e.length-2);n=e[e.length-2]*t,r=e[e.length-1]*t}}else{let t=Ay(e);n=Math.sqrt(t),r=Math.sqrt(t)}return[n,r]}var vb=class extends lb{constructor(e){if(super(),e.scale<0)throw new K(`scale must be a positive float. Got: ${e.scale}`);this.scale=e.scale==null?1:e.scale,this.mode=e.mode==null?`fanIn`:e.mode,sb(this.mode),this.distribution=e.distribution==null?`normal`:e.distribution,cb(this.distribution),this.seed=e.seed}apply(e,t){let n=_b(e),r=n[0],i=n[1],a=this.scale;if(this.mode===`fanIn`?a/=Math.max(1,r):this.mode===`fanOut`?a/=Math.max(1,i):a/=Math.max(1,(r+i)/2),this.distribution===`normal`){let n=Math.sqrt(a);if(t||=`float32`,t!==`float32`&&t!==`int32`)throw new Lv(`${this.getClassName()} does not support dType ${t}.`);return zf(e,0,n,t,this.seed)}{let n=Math.sqrt(3*a);return Td(e,-n,n,t,this.seed)}}getConfig(){return{scale:this.scale,mode:this.mode,distribution:this.distribution,seed:this.seed}}};vb.className=`VarianceScaling`,G(vb);var yb=class extends vb{constructor(e){super({scale:1,mode:`fanAvg`,distribution:`uniform`,seed:e==null?null:e.seed})}getClassName(){return vb.className}};yb.className=`GlorotUniform`,G(yb);var bb=class extends vb{constructor(e){super({scale:1,mode:`fanAvg`,distribution:`normal`,seed:e==null?null:e.seed})}getClassName(){return vb.className}};bb.className=`GlorotNormal`,G(bb);var xb=class extends vb{constructor(e){super({scale:2,mode:`fanIn`,distribution:`normal`,seed:e==null?null:e.seed})}getClassName(){return vb.className}};xb.className=`HeNormal`,G(xb);var Sb=class extends vb{constructor(e){super({scale:2,mode:`fanIn`,distribution:`uniform`,seed:e==null?null:e.seed})}getClassName(){return vb.className}};Sb.className=`HeUniform`,G(Sb);var Cb=class extends vb{constructor(e){super({scale:1,mode:`fanIn`,distribution:`normal`,seed:e==null?null:e.seed})}getClassName(){return vb.className}};Cb.className=`LeCunNormal`,G(Cb);var wb=class extends vb{constructor(e){super({scale:1,mode:`fanIn`,distribution:`uniform`,seed:e==null?null:e.seed})}getClassName(){return vb.className}};wb.className=`LeCunUniform`,G(wb);var Tb=class extends lb{constructor(e){super(),this.DEFAULT_GAIN=1,this.ELEMENTS_WARN_SLOW=2e3,this.gain=e.gain==null?this.DEFAULT_GAIN:e.gain,this.seed=e.seed}apply(e,t){return F(()=>{if(e.length<2)throw new Lv(`Shape must be at least 2D.`);if(t!==`int32`&&t!==`float32`&&t!==void 0)throw TypeError(`Unsupported data type ${t}.`);t=t;let n=C(e.slice(0,-1)),r=e[e.length-1],i=n*r;i>this.ELEMENTS_WARN_SLOW&&console.warn(`Orthogonal initializer is being called on a matrix with more than ${this.ELEMENTS_WARN_SLOW} (${i}) elements: Slowness may result.`);let a=Jy([Math.max(r,n),Math.min(r,n)],0,1,t,this.seed),o=jm.qr(a,!1),s=o[0],c=o[1].flatten().stridedSlice([0],[Math.min(r,n)*Math.min(r,n)],[Math.min(r,n)+1]);return s=z(s,c.sign()),n<r&&(s=s.transpose()),z(H(this.gain),s.reshape(e))})}getConfig(){return{gain:this.gain,seed:this.seed}}};Tb.className=`Orthogonal`,G(Tb);var Eb={constant:`Constant`,glorotNormal:`GlorotNormal`,glorotUniform:`GlorotUniform`,heNormal:`HeNormal`,heUniform:`HeUniform`,identity:`Identity`,leCunNormal:`LeCunNormal`,leCunUniform:`LeCunUniform`,ones:`Ones`,orthogonal:`Orthogonal`,randomNormal:`RandomNormal`,randomUniform:`RandomUniform`,truncatedNormal:`TruncatedNormal`,varianceScaling:`VarianceScaling`,zeros:`Zeros`};function Db(e,t={}){return Xv(e,Im.getMap().classNameMap,t,`initializer`)}function Ob(e){return Jv(e)}function kb(e){if(typeof e==`string`){let t=e in Eb?Eb[e]:e;if(t===`GlorotNormal`)return new bb;if(t===`GlorotUniform`)return new yb;if(t===`HeNormal`)return new xb;if(t===`HeUniform`)return new Sb;if(t===`LeCunNormal`)return new Cb;if(t===`LeCunUniform`)return new wb;{let e={};return e.className=t,e.config={},Db(e)}}return e instanceof lb?e:Db(e)}function Ab(e){return Array.isArray(e)&&Array.isArray(e[0])}function jb(e){return e.length===0?[]:Array.isArray(e[0])?e:[e]}function q(e){let t;if(Array.isArray(e)){if(e.length!==1)throw new K(`Expected Tensor length to be 1; got ${e.length}`);t=e[0]}else t=e;return t}function Mb(e){if(Array.isArray(e)&&Array.isArray(e[0])){if(e.length===1)return e=e,e[0];throw new K(`Expected exactly 1 Shape; got ${e.length}`)}return e}function Nb(e){let t=0;for(let n of e)n.shape.length===0?t+=1:t+=n.shape.reduce((e,t)=>e*t);return t}var Pb=`Variable`,Fb=class{constructor(e,t=`float32`,n=Pb,r=!0,i=null){this.dtype=t??`float32`,this.shape=e.shape,this.id=cy(),n??=Pb,this.originalName=Ty(n),this.name=Ey(this.originalName),this.trainable_=r,this.constraint=i,this.val=Kf(e,this.trainable_,this.name,this.dtype)}read(){return this.assertNotDisposed(),this.val}write(e){return this.assertNotDisposed(),Ib(this.val,e),this.val.id!==e.id&&(this.val.assign(e),this.constraint!=null&&this.val.assign(this.constraint.apply(this.val))),this}dispose(){this.assertNotDisposed(),this.val.dispose()}assertNotDisposed(){if(this.val.isDisposed)throw Error(`LayersVariable ${this.name} is already disposed.`)}get trainable(){return this.trainable_}set trainable(e){this.trainable_=e,this.val.trainable=e}};function Ib(e,t){if(e.shape.toString()!==t.shape.toString())throw Error(`Shape mismatch: `+JSON.stringify(e.shape)+` vs. `+JSON.stringify(t.shape))}function Lb(e){return e.map(e=>e.read())}function Rb(e){e.forEach(e=>{e[0].write(e[1])})}var zb=class{constructor(e){this.dtype=e.dtype,this.shape=e.shape,this.ndim=e.shape==null?e.ndim:e.shape.length,this.maxNDim=e.maxNDim,this.minNDim=e.minNDim,this.axes=e.axes||{}}},Bb=class{constructor(e,t,n,r,i,a,o){this.dtype=e,this.shape=t,this.sourceLayer=n,this.inputs=r,this.callArgs=i,this.outputTensorIndex=o,this.id=cy(),a!=null&&(this.originalName=Ty(a),this.name=Ey(this.originalName)),this.rank=t.length}},Vb=0,Hb=class{constructor(e,t){this.callArgs=t,this.id=Vb++,this.outboundLayer=e.outboundLayer,this.inboundLayers=e.inboundLayers,this.nodeIndices=e.nodeIndices,this.tensorIndices=e.tensorIndices,this.inputTensors=e.inputTensors,this.outputTensors=e.outputTensors,this.inputMasks=e.inputMasks,this.outputMasks=e.outputMasks,this.inputShapes=e.inputShapes,this.outputShapes=e.outputShapes;for(let t of e.inboundLayers)t?.outboundNodes.push(this);e.outboundLayer.inboundNodes.push(this)}getConfig(){let e=[];for(let t of this.inboundLayers)t==null?e.push(null):e.push(t.name);return{outboundLayer:this.outboundLayer?this.outboundLayer.name:null,inboundLayers:e,nodeIndices:this.nodeIndices,tensorIndices:this.tensorIndices}}},Ub=0,Wb=class extends Fm{constructor(e={}){super(),this._callHook=null,this._addedWeightNames=[],this._stateful=!1,this.id=Ub++,this.activityRegularizer=null,this.inputSpec=null,this.supportsMasking=!1,this._trainableWeights=[],this._nonTrainableWeights=[],this._losses=[],this._updates=[],this._built=!1,this.inboundNodes=[],this.outboundNodes=[];let t=e.name;if(!t){let e=this.getClassName();t=Gv(e)+`_`+uy(e)}if(this.name=t,this.trainable_=e.trainable==null||e.trainable,e.inputShape!=null||e.batchInputShape!=null){let t;if(e.batchInputShape!=null)t=e.batchInputShape;else if(e.inputShape!=null){let n=null;e.batchSize!=null&&(n=e.batchSize),t=[n].concat(e.inputShape)}this.batchInputShape=t;let n=e.dtype;n??=e.inputDType,n??=`float32`,this.dtype=n}this.initialWeights=e.weights==null?null:e.weights,this._refCount=null,this.fastWeightInitDuringBuild=!1}static nodeKey(e,t){return e.name+`_ib-`+t.toString()}getNodeAtIndex(e,t){if(this.inboundNodes.length===0)throw new Iv(`The layer has never been called and thus has no defined ${t}.`);if(this.inboundNodes.length<=e)throw new K(`Asked to get ${t} at node ${e}, but the layer has only ${this.inboundNodes.length} inbound nodes.`);return this.inboundNodes[e]}getInputAt(e){return Uv(this.getNodeAtIndex(e,`input`).inputTensors)}getOutputAt(e){return Uv(this.getNodeAtIndex(e,`output`).outputTensors)}get input(){if(this.inboundNodes.length>1)throw new Fv(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer input" is ill-defined. Use \`getInputAt(nodeIndex)\` instead.`);if(this.inboundNodes.length===0)throw new Fv(`Layer ${this.name} is not connected, no input to return.`);return Uv(this.getNodeAtIndex(0,`input`).inputTensors)}get output(){if(this.inboundNodes.length===0)throw new Fv(`Layer ${this.name} has no inbound nodes.`);if(this.inboundNodes.length>1)throw new Fv(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer output" is ill-defined. Use \`getOutputAt(nodeIndex)\` instead.`);return Uv(this.getNodeAtIndex(0,`output`).outputTensors)}get losses(){return this._losses}calculateLosses(){return this.losses.map(e=>e())}get updates(){return this._updates}get built(){return this._built}set built(e){this._built=e}get trainable(){return this.trainable_}set trainable(e){this._trainableWeights.forEach(t=>t.trainable=e),this.trainable_=e}get trainableWeights(){return this.trainable_?this._trainableWeights.filter(e=>e.trainable):[]}set trainableWeights(e){this._trainableWeights=e}get nonTrainableWeights(){return this.trainable?this._trainableWeights.filter(e=>!e.trainable).concat(this._nonTrainableWeights):this._trainableWeights.concat(this._nonTrainableWeights)}set nonTrainableWeights(e){this._nonTrainableWeights=e}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}get stateful(){return this._stateful}resetStates(){if(!this.stateful)throw Error(`Cannot call the resetStates() method of a non-stateful Layer object.`)}assertInputCompatibility(e){let t=Wv(e);if(this.inputSpec==null||this.inputSpec.length===0)return;let n=Wv(this.inputSpec);if(t.length!==n.length)throw new K(`Layer ${this.name} expects ${n.length} inputs, but it received ${t.length} input tensors. Input received: ${e}`);for(let e=0;e<t.length;e++){let r=t[e],i=n[e];if(i==null)continue;let a=r.rank;if(i.ndim!=null&&a!==i.ndim)throw new K(`Input ${e} is incompatible with layer ${this.name}: expected ndim=${i.ndim}, found ndim=${a}`);if(i.maxNDim!=null&&a>i.maxNDim)throw new K(`Input ${e} is incompatible with layer ${this.name}: expected max_ndim=${i.maxNDim}, found ndim=${a}`);if(i.minNDim!=null&&a<i.minNDim)throw new K(`Input ${e} is incompatible with layer ${this.name}: expected min_ndim=${i.minNDim}, found ndim=${a}.`);if(i.dtype!=null&&r.dtype!==i.dtype)throw new K(`Input ${e} is incompatible with layer ${this.name} : expected dtype=${i.dtype}, found dtype=${r.dtype}.`);if(i.axes){let t=r.shape;for(let n in i.axes){let r=Number(n),a=i.axes[n],o=r>=0?t[r]:t[t.length+r];if(a!=null&&[a,null].indexOf(o)===-1)throw new K(`Input ${e} is incompatible with layer ${this.name}: expected axis ${r} of input shape to have value ${a} but got shape ${t}.`)}}if(i.shape!=null)for(let t=0;t<i.shape.length;++t){let n=i.shape[t],a=r.shape[t];if(n!=null&&a!=null&&n!==a)throw new K(`Input ${e} is incompatible with layer ${this.name}: expected shape=${i.shape}, found shape=${r.shape}.`)}}}call(e,t){return e}invokeCallHook(e,t){this._callHook!=null&&this._callHook(e,t)}setCallHook(e){this._callHook=e}clearCallHook(){this._callHook=null}apply(e,t){t||={},this.assertNotDisposed();let n=Wv(e),r=Jb(e),i=Yb(e);if(r===i)throw new K(`Arguments to apply() must be all SymbolicTensors or all Tensors`);return Cy(this.name,()=>{if(!this.built){this.assertInputCompatibility(e);let t=[];for(let n of Wv(e))t.push(n.shape);this.build(Uv(t)),this.built=!0,this.initialWeights&&this.setWeights(this.initialWeights),this._refCount===null&&i&&(this._refCount=1)}if(this.assertInputCompatibility(e),i){let r=this.call(e,t);this.supportsMasking&&this.setMaskMetadata(e,r);let i=Wv(r),a=[];for(let e of i)n.indexOf(e)!==-1&&(e=e.clone()),a.push(e);if(r=Uv(a),this.activityRegularizer!=null)throw new Lv(`Layer invocation in the presence of activity regularizer(s) is not supported yet.`);return r}{let n=Gb(e),r=this.computeOutputShape(n),i,a=Kb(e);if(this.warnOnIncompatibleInputShape(Array.isArray(e)?n[0]:n),i=r!=null&&r.length>0&&Array.isArray(r[0])?r.map((n,r)=>new Bb(a,n,this,Wv(e),t,this.name,r)):new Bb(a,r,this,Wv(e),t,this.name),this.addInboundNode(e,i,null,null,n,r,t),this._refCount++,this.activityRegularizer!=null)throw new Lv(`Layer invocation in the presence of activity regularizer(s) is not supported yet.`);return i}})}warnOnIncompatibleInputShape(e){if(this.batchInputShape!=null){if(e.length!==this.batchInputShape.length)console.warn(`The rank of the input tensor provided (shape: ${JSON.stringify(e)}) does not match that of the batchInputShape (${JSON.stringify(this.batchInputShape)}) of the layer ${this.name}`);else{let t=!1;this.batchInputShape.forEach((n,r)=>{n!=null&&e[r]!=null&&e[r]!==n&&(t=!0)}),t&&console.warn(`The shape of the input tensor (${JSON.stringify(e)}) does not match the expectation of layer ${this.name}: ${JSON.stringify(this.batchInputShape)}`)}}}get outputShape(){if(this.inboundNodes==null||this.inboundNodes.length===0)throw new Fv(`The layer ${this.name} has never been called and thus has no defined output shape.`);let e=[];for(let t of this.inboundNodes){let n=JSON.stringify(t.outputShapes);e.indexOf(n)===-1&&e.push(n)}if(e.length===1){let e=this.inboundNodes[0].outputShapes;return Array.isArray(e)&&Array.isArray(e[0])&&e.length===1?e[0]:e}throw new Fv(`The layer ${this.name} has multiple inbound nodes with different output shapes. Hence the notion of "output shape" is ill-defined for the layer.`)}countParams(){if(!this.built)throw new Iv(`You tried to call countParams() on ${this.name}, but the layer is not built yet. Build it first by calling build(batchInputShape).`);return Nb(this.weights)}build(e){this.built=!0}getWeights(e=!1){return Lb(e?this.trainableWeights:this.weights)}setWeights(e){F(()=>{let t=this.weights;if(t.length!==e.length)throw new K(`You called setWeights(weights) on layer "${this.name}" with a weight list of length ${e.length}, but the layer was expecting ${t.length} weights. Provided weights: ${e}...`);if(t.length===0)return;let n=[],r=Lb(t);for(let i=0;i<r.length;++i){let a=r[i],o=t[i],s=e[i];if(!w(a.shape,s.shape))throw new K(`Layer weight shape ${a.shape} not compatible with provided weight shape ${s.shape}`);n.push([o,s])}Rb(n)})}addWeight(e,t,n,r,i,a,o,s){if(this._addedWeightNames.indexOf(e)!==-1)throw new K(`Duplicate weight name ${e} for layer ${this.name}`);this._addedWeightNames.push(e),n??=`float32`,this.fastWeightInitDuringBuild&&(r=s==null?kb(`zeros`):s());let c=r.apply(t,n),l=new Fb(c,n,e,a,o);return c.dispose(),i!=null&&this.addLoss(()=>i.apply(l.read())),a??=!0,a?this._trainableWeights.push(l):this._nonTrainableWeights.push(l),l}setFastWeightInitDuringBuild(e){this.fastWeightInitDuringBuild=e}addLoss(e){e==null||Array.isArray(e)&&e.length===0||(e=Wv(e),this._losses!==void 0&&this._losses!==null&&this.losses.push(...e))}computeOutputShape(e){return e}computeMask(e,t){if(!this.supportsMasking){if(t!=null){if(Array.isArray(t))t.forEach(e=>{if(e!=null)throw TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`)});else throw TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`)}return null}return t}setMaskMetadata(e,t,n){if(!this.supportsMasking)return;let r=this.computeMask(e,n),i=Wv(t),a=Wv(r);if(i.length!==a.length)throw Error(`${this.name} outputs ${i.length} tensors but ${i.length} masks for those tensors`);for(let e=0;e<i.length;e++)i[e].kerasMask=a[e]}addInboundNode(e,t,n,r,i,a,o=null){let s=Wv(e);t=Wv(t),n=Wv(n),r=Wv(r),i=jb(i),a=jb(a);let c=[],l=[],u=[];for(let e of s)c.push(e.sourceLayer),l.push(e.nodeIndex),u.push(e.tensorIndex);new Hb({outboundLayer:this,inboundLayers:c,nodeIndices:l,tensorIndices:u,inputTensors:s,outputTensors:t,inputMasks:n,outputMasks:r,inputShapes:i,outputShapes:a},o);for(let e=0;e<t.length;e++)t[e].sourceLayer=this,t[e].nodeIndex=this.inboundNodes.length-1,t[e].tensorIndex=e}getConfig(){let e={name:this.name,trainable:this.trainable};return this.batchInputShape!=null&&(e.batchInputShape=this.batchInputShape),this.dtype!=null&&(e.dtype=this.dtype),e}disposeWeights(){return this.weights.forEach(e=>e.dispose()),this.weights.length}assertNotDisposed(){if(this._refCount===0)throw Error(`Layer '${this.name}' is already disposed.`)}dispose(){if(!this.built)throw Error(`Cannot dispose Layer ${this.name} because it has not been built yet.`);if(this._refCount===null)throw Error(`Cannot dispose Layer ${this.name} because it has not been used yet.`);this.assertNotDisposed();let e=0;return--this._refCount===0&&(e=this.disposeWeights()),{refCountAfterDispose:this._refCount,numDisposedVariables:e}}};function Gb(e){e=Wv(e);let t=[];for(let n of e)t.push(n.shape);return Uv(t)}function Kb(e){return`float32`}function qb(e,t,n){if((t==null||n!=null&&n>0)&&(t=e.sourceLayer,n=e.nodeIndex),t.inboundNodes.length===0)return[e];{let e=t.inboundNodes[n];if(e.inboundLayers.length===0)return e.inputTensors;{let t=[];for(let n=0;n<e.inboundLayers.length;n++){let r=e.inputTensors[n],i=e.inboundLayers[n],a=e.nodeIndices[n],o=qb(r,i,a);for(let e of o)t.indexOf(e)===-1&&t.push(e)}return t}}}function Jb(e){let t=!0;for(let n of Wv(e))if(!(n instanceof Bb)){t=!1;break}return t}function Yb(e){let t=!0;for(let n of Wv(e))if(n instanceof Bb){t=!1;break}return t}var Xb=class extends Wb{constructor(e){if(super({dtype:e.dtype,name:e.name==null?uy(`input`).toString():e.name}),e.batchSize??=null,e.sparse??=!1,this.trainable=!1,this.built=!0,this.sparse=e.sparse,e.inputShape!=null&&e.batchInputShape!=null)throw new K(`Only provide the inputShape OR batchInputShape argument to inputLayer, not both at the same time.`);let t=e.batchInputShape;if(t==null){if(e.inputShape==null)throw new K("An InputLayer should be passed either a `batchInputShape` or an `inputShape`.");t=[e.batchSize].concat(e.inputShape)}else if(e.batchSize!=null)throw new K(`Cannot specify batchSize if batchInputShape is specified when creating an InputLayer.`);let n=e.dtype||`float32`;this.batchInputShape=t,this.dtype=n,this.inputSpec=[{shape:t}];let r=new Bb(this.dtype,this.batchInputShape,this,[],{},this.name);r.nodeIndex=0,r.tensorIndex=0,new Hb({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:[r],outputTensors:[r],inputMasks:[null],outputMasks:[null],inputShapes:[t],outputShapes:[t]})}apply(e,t){throw new K(`Cannot pass any input to an InputLayer's apply() method. InputLayer name: ${this.name}`)}dispose(){return{refCountAfterDispose:this._refCount,numDisposedVariables:0}}getConfig(){return{batchInputShape:this.batchInputShape,dtype:this.dtype,sparse:this.sparse,name:this.name}}};Xb.className=`InputLayer`,G(Xb);function Zb(e){if(e.batchShape==null&&e.shape==null)throw Error("Please provide to Input either a `shape` or a `batchShape` argument. Note that `shape` does not include the batch dimension.");if(e.batchShape!=null&&e.shape!=null)throw new K("Please provide either a `shape` or `batchShape` argument to Input, but not both.");let t=e.batchShape;e.shape!=null&&t==null&&(t=[null].concat(e.shape));let n=e.dtype;return n??=`float32`,new Xb({batchInputShape:t,name:e.name,dtype:n,sparse:e.sparse}).inboundNodes[0].outputTensors[0]}function Qb(e,t){if(e.dtype==null||e.dtype===t.dtype)return t;try{return I(t,e.dtype)}catch{throw new K(`The dtype of the feed (${t.dtype}) can not be cast to the dtype of the key '${e.name}' (${e.dtype}).`)}}var $b=class e{constructor(t){if(this.id2Value={},this.id2Mask={},this.name2Id={},t instanceof e)for(let e in t.id2Value)this.id2Value[e]=t.id2Value[e],e in t.id2Mask&&(this.id2Mask[e]=t.id2Mask[e]);else{if(t==null)return;for(let e of t)this.add(e.key,e.value)}}add(e,t,n){if(this.id2Value[e.id]==null)this.id2Value[e.id]=Qb(e,t),this.name2Id[e.name]=e.id,n!=null&&(this.id2Mask[e.id]=n);else throw new K(`Duplicate key: name=${e.name}, id=${e.id}`);return this}addFeed(e){this.add(e.key,e.value)}hasKey(e){return this.id2Value[e.id]!=null}names(){return Object.keys(this.name2Id)}getValue(e){if(e instanceof Bb){if(this.id2Value[e.id]==null)throw new K(`Nonexistent key: ${e.name}`);return this.id2Value[e.id]}{let t=this.name2Id[e];if(t==null)throw new K(`Feed dict has no SymbolicTensor name: ${e}`);return this.id2Value[t]}}getMask(e){if(e instanceof Bb){if(this.id2Value[e.id]==null)throw new K(`Nonexistent key: ${e.name}`);return this.id2Mask[e.id]}{let t=this.name2Id[e];if(t==null)throw new K(`Feed dict has no SymbolicTensor name: ${e}`);return this.id2Mask[t]}}disposeMasks(){this.id2Mask!=null&&ya(this.id2Mask)}},ex=new zv,tx=new zv;function nx(e){ex?.setMaxEntries(e),tx?.setMaxEntries(e)}function rx(e,t,n,r){let i=n!=null&&n.training,a=Array.isArray(e),o=a?e:[e],s=o.map(e=>e.name),c=[],l=t.names();for(let e of s)l.indexOf(e)===-1?c.push(null):c.push(t.getValue(e));r!=null&&(r.maxNumTensors=-1/0,r.minNumTensors=1/0);let u=s.join(`,`)+`|`+t.names().sort().join(`,`),d=ex.get(u),f;if(d==null){let e=ix(o,t);d=e.sorted,f=e.recipientCounts,ex.put(u,d),tx.put(u,f)}f={},i||Object.assign(f,tx.get(u));let p=new $b(t);for(let e=0;e<d.length;++e){if(r!=null){let e=va().numTensors;e>r.maxNumTensors&&(r.maxNumTensors=e),e<r.minNumTensors&&(r.minNumTensors=e)}let a=d[e],o=a.sourceLayer;if(o instanceof Xb)continue;let l=[],u=[],m=[],h=!1;for(let e of a.inputs){let n=p.getValue(e),r=p.getMask(e);l.push(n),u.push(r),r!=null&&(h=!0),i||(f[e.name]--,f[e.name]===0&&!t.hasKey(e)&&s.indexOf(e.name)===-1&&!n.isDisposed&&e.sourceLayer.stateful!==!0&&m.push(n))}h&&(n||={},n.mask=u[0]);let g=Wv(o.apply(l,n)),_=null;o.supportsMasking&&(_=o.computeMask(l,u));let v=sx(a),y=Array.isArray(v)?v:[v];for(let e=0;e<y.length;++e){p.hasKey(y[e])||p.add(y[e],g[e],Array.isArray(_)?_[0]:_);let t=s.indexOf(y[e].name);t!==-1&&(c[t]=g[e])}i||ya(m)}return p.disposeMasks(),a?c:c[0]}function ix(e,t){b(e!=null&&e.length>0,()=>`Expected at least one fetch, got none`);let n=[],r={};if(e.length===1){let i=ox(e[0],t);n=i.sorted,r=i.recipientMap}else{let i=new Set;for(let a of e){let{sorted:e,recipientMap:o}=ox(a,t);for(let t of e)i.has(t.name)||(n.push(t),i.add(t.name));for(let e in o)r[e]??(r[e]=new Set),o[e].forEach(t=>r[e].add(t))}}return{sorted:n,recipientCounts:ax(r)}}function ax(e){let t={};for(let n in e)t[n]=e[n].size;return t}function ox(e,t){let n=new Set,r=[],i={};for(let e of t.names())n.add(e);let a=[],o=[];for(a.push(e);a.length>0;){let e=a[a.length-1];if(n.has(e.name)){a.pop();continue}let t=o[o.length-1]===a.length-1;if(e.inputs.length===0||t)a.pop(),r.push(e),n.add(e.name),t&&o.pop();else{o.push(a.length-1);for(let t of e.inputs)i[t.name]??(i[t.name]=new Set),i[t.name].add(e.name),!n.has(t.name)&&a.push(t)}}return{sorted:r,recipientMap:i}}function sx(e){let t;if(e.sourceLayer.inboundNodes.length===1)t=e.sourceLayer.output;else{let n=null;for(let t=0;t<e.sourceLayer.inboundNodes.length;++t)for(let r of e.sourceLayer.inboundNodes[t].outputTensors)if(r.id===e.id){n=t;break}t=e.sourceLayer.getOutputAt(n)}return t}A().registerFlag(`TOPOLOGICAL_SORT_CACHE_MAX_ENTRIES`,()=>100,nx);function cx(e,t){return F(()=>gl(U(z(e,e),t,!0)))}var lx=class extends Fm{getConfig(){return{}}},ux=class extends lx{constructor(e){super(),this.defaultMaxValue=2,this.defaultAxis=0,this.maxValue=e.maxValue==null?this.defaultMaxValue:e.maxValue,this.axis=e.axis==null?this.defaultAxis:e.axis}apply(e){return F(()=>{let t=cx(e,this.axis);return z(e,R(Qs(t,0,this.maxValue),L(Fy(),t)))})}getConfig(){return{maxValue:this.maxValue,axis:this.axis}}};ux.className=`MaxNorm`,G(ux);var dx=class extends lx{constructor(e){super(),this.defaultAxis=0,this.axis=e.axis==null?this.defaultAxis:e.axis}apply(e){return F(()=>R(e,L(Fy(),cx(e,this.axis))))}getConfig(){return{axis:this.axis}}};dx.className=`UnitNorm`,G(dx);var fx=class extends lx{apply(e){return Md(e)}};fx.className=`NonNeg`,G(fx);var px=class extends lx{constructor(e){super(),this.defaultMinValue=0,this.defaultMaxValue=1,this.defaultRate=1,this.defaultAxis=0,this.minValue=e.minValue==null?this.defaultMinValue:e.minValue,this.maxValue=e.maxValue==null?this.defaultMaxValue:e.maxValue,this.rate=e.rate==null?this.defaultRate:e.rate,this.axis=e.axis==null?this.defaultAxis:e.axis}apply(e){return F(()=>{let t=cx(e,this.axis);return z(e,R(L(z(this.rate,Qs(t,this.minValue,this.maxValue)),z(1-this.rate,t)),L(Fy(),t)))})}getConfig(){return{minValue:this.minValue,maxValue:this.maxValue,rate:this.rate,axis:this.axis}}};px.className=`MinMaxNorm`,G(px);var mx={maxNorm:`MaxNorm`,minMaxNorm:`MinMaxNorm`,nonNeg:`NonNeg`,unitNorm:`UnitNorm`};function hx(e){return Jv(e)}function gx(e,t={}){return Xv(e,Im.getMap().classNameMap,t,`constraint`)}function _x(e){return e==null?null:typeof e==`string`?gx({className:e in mx?mx[e]:e,config:{}}):e instanceof lx?e:gx(e)}async function vx(e){if(e==null)return;let t=[],n=[],r=[];for(let i in e){let a=e[i];if(typeof a!=`number`){let e=a;t.push(e.data()),n.push(i),r.push(e)}}if(t.length>0){let i=await Promise.all(t);for(let t=0;t<i.length;++t)e[n[t]]=i[t][0];ya(r)}}function yx(e){if(e!=null)for(let t in e){let n=e[t];typeof n!=`number`&&n.dispose()}}var bx;(function(e){e[e.SILENT=0]=`SILENT`,e[e.VERBOSE=1]=`VERBOSE`})(bx||={});var xx=class{constructor(){this.validationData=null}setParams(e){this.params=e}async onEpochBegin(e,t){}async onEpochEnd(e,t){}async onBatchBegin(e,t){}async onBatchEnd(e,t){}async onTrainBegin(e){}async onTrainEnd(e){}setModel(e){}},Sx=class{constructor(e,t=10){e??=[],this.callbacks=e,this.queueLength=t}append(e){this.callbacks.push(e)}setParams(e){for(let t of this.callbacks)t.setParams(e)}setModel(e){for(let t of this.callbacks)t.setModel(e)}async onEpochBegin(e,t){t??={};for(let n of this.callbacks)await n.onEpochBegin(e,t)}async onEpochEnd(e,t){t??={};for(let n of this.callbacks)await n.onEpochEnd(e,t)}async onBatchBegin(e,t){t??={};for(let n of this.callbacks)await n.onBatchBegin(e,t)}async onBatchEnd(e,t){t??={};for(let n of this.callbacks)await n.onBatchEnd(e,t)}async onTrainBegin(e){e??={};for(let t of this.callbacks)await t.onTrainBegin(e)}async onTrainEnd(e){e??={};for(let t of this.callbacks)await t.onTrainEnd(e)}},Cx=class extends xx{constructor(){super()}async onEpochBegin(e){this.seen=0,this.totals={}}async onBatchEnd(e,t){t??={};let n=t.size==null?0:t.size;this.seen+=n;for(let e in t){let r=t[e];if(typeof r==`number`)this.totals.hasOwnProperty(e)||(this.totals[e]=0),this.totals[e]=this.totals[e]+r*n;else{let t;e in this.totals?t=this.totals[e]:this.totals[e]=0;let i=F(()=>L(this.totals[e],z(r,n)));this.totals[e]=i,t?.dispose()}}}async onEpochEnd(e,t){if(t!=null)for(let e of this.params.metrics)this.totals[e]!=null&&(typeof this.totals[e]==`number`?t[e]=this.totals[e]/this.seen:F(()=>{let n=z(R(1,this.seen),this.totals[e]);t[e]=n,this.totals[e].dispose(),ba(t[e])}))}},wx=class extends xx{async onTrainBegin(e){this.epoch=[],this.history={}}async onEpochEnd(e,t){t??={},this.epoch.push(e);for(let e in t)this.history[e]??(this.history[e]=[]),this.history[e].push(t[e])}async syncData(){let e=[],t=[],n=[];for(let r in this.history){let i=this.history[r];for(let a=0;a<i.length;++a)if(typeof i[a]!=`number`){let o=i[a];e.push(o.data()),t.push(r),n.push(a)}}let r=await Promise.all(e);for(let e=0;e<r.length;++e)this.history[t[e]][n[e]].dispose(),this.history[t[e]][n[e]]=r[e][0]}},Tx=class extends xx{constructor(e,t){if(super(),this.currentEpoch=0,this.nowFunc=e.nowFunc,this.nextFrameFunc=e.nextFrameFunc||_h,this.yieldEvery=t||`auto`,this.yieldEvery===`auto`&&(this.yieldEvery=125),this.yieldEvery===`never`&&e.onYield!=null)throw Error("yieldEvery is `never` but you provided an `onYield` callback. Either change `yieldEvery` or remove the callback");fe(this.yieldEvery)&&(this.maybeWait=ay(this.maybeWait.bind(this),this.yieldEvery,this.nowFunc)),this.trainBegin=e.onTrainBegin,this.trainEnd=e.onTrainEnd,this.epochBegin=e.onEpochBegin,this.epochEnd=e.onEpochEnd,this.batchBegin=e.onBatchBegin,this.batchEnd=e.onBatchEnd,this.yield=e.onYield}async maybeWait(e,t,n){let r=[];this.yield!=null&&(await vx(n),r.push(this.yield(e,t,n))),r.push(this.nextFrameFunc()),await Promise.all(r)}async onEpochBegin(e,t){this.currentEpoch=e,this.epochBegin!=null&&(await vx(t),await this.epochBegin(e,t))}async onEpochEnd(e,t){let n=[];this.epochEnd!=null&&(await vx(t),n.push(this.epochEnd(e,t))),this.yieldEvery===`epoch`&&n.push(this.nextFrameFunc()),await Promise.all(n)}async onBatchBegin(e,t){this.batchBegin!=null&&(await vx(t),await this.batchBegin(e,t))}async onBatchEnd(e,t){let n=[];this.batchEnd!=null&&(await vx(t),n.push(this.batchEnd(e,t))),this.yieldEvery===`batch`?n.push(this.nextFrameFunc()):fe(this.yieldEvery)&&n.push(this.maybeWait(this.currentEpoch,e,t)),await Promise.all(n)}async onTrainBegin(e){this.trainBegin!=null&&(await vx(e),await this.trainBegin(e))}async onTrainEnd(e){this.trainEnd!=null&&(await vx(e),await this.trainEnd(e))}};function Ex(e,t){return e??={},e instanceof xx?[e]:Array.isArray(e)&&e[0]instanceof xx?e:Wv(e).map(e=>new Tx(e,t))}var Dx=class e{constructor(){}static registerCallbackConstructor(t,n){b(t>=0&&Number.isInteger(t),()=>`Verbosity level is expected to be an integer >= 0, but got ${t}`),e.checkForDuplicate(n),e.constructors[t]??(e.constructors[t]=[]),e.constructors[t].push(n)}static checkForDuplicate(t){for(let n in e.constructors)e.constructors[+n].forEach(e=>{if(e===t)throw new K(`Duplicate callback constructor.`)})}static clear(){e.constructors={}}static createCallbacks(t){let n=[];for(let r in e.constructors){let i=+r;t>=i&&n.push(...e.constructors[i])}return n.map(e=>new e)}};Dx.constructors={};function Ox(e,t,n,r,i,a,o,s,c){let l=new wx,u=[new Cx,...Dx.createCallbacks(t)];e!=null&&u.push(...e),u.push(l);let d=new Sx(u);return d.setParams({epochs:n,initialEpoch:r,samples:i,steps:a,batchSize:o,verbose:t,doValidation:s,metrics:c}),{callbackList:d,history:l}}function kx(e,t={},n=!1){return Xv(e,Im.getMap().classNameMap,t,`layer`,n)}function Ax(e,t){return F(()=>{e.dtype!==`float32`&&(e=I(e,`float32`));let n=U(Zy(e),t,!0),r=gl(Fu(n,Xs(n.shape,Fy())));return R(e,r)})}function jx(e,t){return F(()=>Lu(Zy(W(t,e)),-1))}function Mx(e,t){return F(()=>Lu(Oo(W(t,e)),-1))}function Nx(e,t){return F(()=>z(100,Lu(Oo(R(W(e,t),Qs(Oo(e),Fy(),Number.MAX_VALUE))),-1)))}function Px(e,t){return F(()=>Lu(Zy(W(ou(L(1,Qs(t,Fy(),Number.MAX_VALUE))),ou(L(1,Qs(e,Fy(),Number.MAX_VALUE))))),-1))}function Fx(e,t){return F(()=>Lu(Zy(Fu(0,W(1,z(e,t)))),-1))}function Ix(e,t){return F(()=>Lu(Fu(0,W(1,z(e,t))),-1))}function Lx(e,t){return F(()=>{let n=U(z(e,t),-1);return Fu(0,L(1,W(ul(z(W(1,e),t),-1),n)))})}function Rx(e,t){return F(()=>{let n=Math.log(2),r=W(t,e);return Lu(W(L(r,mu(z(-2,r))),n),-1)})}function zx(e,t,n=!1){return F(()=>{if(n)t=sf(t);else{let e=U(t,t.shape.length-1,!0);t=R(t,e)}return t=Qs(t,Fy(),1-Fy()),fu(U(z(I(e,`float32`),ou(t)),t.shape.length-1))})}function Bx(e,t,n=!1){return F(()=>{let r=I(Il(By(e)),`int32`);t=Qs(t,Fy(),1-Fy());let i=t.shape;return zx(B(Zu(r,i[i.length-1]),i),t,n)})}function Vx(e,t){if(!w(e.shape,t.shape))throw new K(`logits and labels must have the same shape, but got shapes ${JSON.stringify(e.shape)} and ${JSON.stringify(t.shape)}`);return F(()=>{let n=Md(t),r=fu(Oo(t));return L(W(n,z(t,e)),cu(El(r)))})}function Hx(e,t){return F(()=>{let n;return n=Qs(t,Fy(),1-Fy()),n=ou(R(n,W(1,n))),Lu(Vx(e,n),-1)})}function Ux(e,t){return F(()=>U(z(e,ou(R(Qs(e,Fy(),1),Qs(t,Fy(),1)))),-1))}function Wx(e,t){return F(()=>Lu(W(t,z(e,ou(L(Fy(),t)))),-1))}function Gx(e,t){return F(()=>fu(U(z(Ax(e,-1),Ax(t,-1)),-1)))}var Kx={meanSquaredError:jx,meanAbsoluteError:Mx,meanAbsolutePercentageError:Nx,meanSquaredLogarithmicError:Px,squaredHinge:Fx,hinge:Ix,categoricalHinge:Lx,logcosh:Rx,categoricalCrossentropy:zx,sparseCategoricalCrossentropy:Bx,binaryCrossentropy:Hx,kullbackLeiblerDivergence:Ux,poisson:Wx,cosineProximity:Gx};function qx(e){if(typeof e==`string`){if(e in Kx)return Kx[e];let t=`Unknown loss ${e}`;throw e.toLowerCase().includes(`softmaxcrossentropy`)&&(t=`Unknown loss ${e}. Use "categoricalCrossentropy" as the string name for tf.losses.softmaxCrossEntropy`),new K(t)}return e}function Jx(e,t){return F(()=>Lu(Bc(e,Ly(Bl(t,z(.5,$u(t))),e.dtype)),-1))}function Yx(e,t){return F(()=>Ly(Bc(Bo(e,-1),Bo(t,-1)),`float32`))}function Xx(e,t){return F(()=>I(U(Cu(Bc(e,1),Bc(t,1))),`float32`))}function Zx(e,t){return F(()=>I(U(Cu(Bc(e,0),Bc(t,1))),`float32`))}function Qx(e,t){return F(()=>{let n=Xx(e,t),r=L(n,Zx(e,t));return I(Hc(Bl(r,0),R(n,r),0),`float32`)})}function $x(e,t){return Hx(e,t)}function eS(e,t){return e.rank===t.rank&&(e=xf(e,[e.rank-1])),t=Bo(t,-1),t.dtype!==e.dtype&&(t=I(t,e.dtype)),I(Bc(e,t),`float32`)}var tS=jx,nS=jx,rS=Mx,iS=Mx,aS=Nx,oS=Nx,sS=zx,cS=Gx,lS=Bx,uS={binaryAccuracy:Jx,categoricalAccuracy:Yx,precision:Qx,categoricalCrossentropy:sS,sparseCategoricalCrossentropy:lS,mse:tS,MSE:nS,mae:rS,MAE:iS,mape:aS,MAPE:oS,cosine:cS};function dS(e){if(typeof e==`string`&&e in uS)return uS[e];if(typeof e!=`string`&&e!=null)return e;throw new K(`Unknown metric ${e}`)}function fS(e){if(Vv(e!==null,`Unknown LossOrMetricFn ${e}`),typeof e==`string`)return e;{let t;for(let n of Object.keys(Kx))if(Kx[n]===e){t=n;break}if(t!==void 0)return t;for(let n of Object.keys(uS))if(uS[n]===e){t=n;break}return t===void 0?e.name:t}}function pS(e){let t={Adagrad:()=>hh.adagrad(.01),Adadelta:()=>hh.adadelta(1,.95,Fy()),Adam:()=>hh.adam(.001,.9,.999,Fy()),Adamax:()=>hh.adamax(.002,.9,.999,Fy(),0),RMSProp:()=>hh.rmsprop(.001,.9,0,Fy()),SGD:()=>hh.sgd(.01)};if(t.adagrad=t.Adagrad,t.adadelta=t.Adadelta,t.adam=t.Adam,t.adamax=t.Adamax,t.rmsprop=t.RMSProp,t.sgd=t.SGD,e in t)return t[e]();throw new K(`Unknown Optimizer ${e}`)}var mS=1048576;function hS(e,t,n=!1){if(typeof e!=`object`||!e||Object.getPrototypeOf(e)!==Object.prototype||!gS(e))throw Error(`User-defined metadata is expected to be a JSON object, but is not.`);if(n){let n=JSON.stringify(e);n.length>1048576&&console.warn(`User-defined metadata of model "${t}" is too large in size (length=${n.length} when serialized). It is not recommended to store such large objects in user-defined metadata. Please make sure its serialized length is <= ${mS}.`)}}function gS(e){if(e===null)return!0;if(typeof e==`object`){if(Object.getPrototypeOf(e)===Object.prototype){let t=Object.keys(e);for(let n of t)if(typeof n!=`string`||!gS(e[n]))return!1;return!0}if(Array.isArray(e)){for(let t of e)if(!gS(t))return!1;return!0}return!1}{let t=typeof e;return t===`string`||t===`number`||t===`boolean`}}function _S(e,t,n,r=console.log){let i=yS(e),a=[`Layer (type)`,`Input Shape`,`Output shape`,`Param #`];i?(t||=90,n||=[.32,.61,.89,1]):(t||=115,n||=[.24,.48,.7,.8,1]),n[n.length-1]<=1&&(n=n.map(e=>Math.floor(t*e)));let o;if(!i){a.push(`Receives inputs`),o=[];for(let t in e.nodesByDepth)o.push(...e.nodesByDepth[t])}r(`_`.repeat(t)),bS(a,n,r),r(`=`.repeat(t));let s=e.layers;for(let e=0;e<s.length;++e)i?xS(s[e],n,r):SS(s[e],n,o,r),r((e===s.length-1?`=`:`_`).repeat(t));e.checkTrainableWeightsConsistency();let c=vS(e),l=Nb(e.nonTrainableWeights);r(`Total params: ${c+l}`),r(`Trainable params: ${c}`),r(`Non-trainable params: ${l}`),r(`_`.repeat(t))}function vS(e){let t;return t=e.collectedTrainableWeights==null?Nb(e.trainableWeights):Nb(e.collectedTrainableWeights),t}function yS(e){let t=!0,n=[],r=[];for(let t in e.nodesByDepth)n.push(e.nodesByDepth[t]);for(let e of n){if(e.length>1||e.length===1&&e[0].inboundLayers.length>1){t=!1;break}r.push(...e)}if(t)for(let n of e.layers){let e=!1;for(let i of n.inboundNodes)if(r.indexOf(i)!==-1){if(e){t=!1;break}e=!0}if(!t)break}return t}function bS(e,t,n=console.log){let r=``;for(let n=0;n<e.length;++n)n>0&&(r=r.slice(0,r.length-1)+` `),r+=e[n],r=r.slice(0,t[n]),r+=` `.repeat(t[n]-r.length);n(r)}function xS(e,t,n){let r,i;try{i=e.inboundNodes.map(e=>JSON.stringify(e.inputShapes)).join(`,`)}catch{i=`multiple`}try{r=JSON.stringify(e.outputShape)}catch{r=`multiple`}bS([`${e.name} (${e.getClassName()})`,i,r,e.countParams().toString()],t,n)}function SS(e,t,n,r){let i,a;try{a=e.inboundNodes.map(e=>JSON.stringify(e.inputShapes)).join(`,`)}catch{a=`multiple`}try{i=JSON.stringify(e.outputShape)}catch{i=`multiple`}let o=[];for(let t of e.inboundNodes)if(!(n!=null&&n.length>0&&n.indexOf(t)===-1))for(let e=0;e<t.inboundLayers.length;++e){let n=t.inboundLayers[e].name,r=t.nodeIndices[e],i=t.tensorIndices[e];o.push(`${n}[${r}][${i}]`)}let s=e.name,c=e.getClassName(),l=o.length===0?``:o[0];bS([`${s} (${c})`,a,i,e.countParams().toString(),l],t,r);for(let e=1;e<o.length;++e)bS([``,``,``,``,o[e]],t,r)}function CS(e,t,n){return(e===`inboundNodes`||e===`outputLayers`||e===`inputLayers`)&&t===0&&typeof n==`string`}function wS(e,t){if(e===null)return null;if(typeof e==`string`)return Kv(e);if(typeof e==`number`||typeof e==`boolean`)return e;if(e instanceof Array){let n=[],r=e.length;for(let i=0;i<r;++i){let r=e[i];CS(t,i,r)?n.push(r):n.push(wS(r,t))}return n}{let t={};for(let n of Object.keys(e)){let r=e[n];if(n===`name`&&typeof r==`string`)t[n]=r;else{let e=Kv(n);t[e]=wS(r,e)}}return t}}function TS(e,t){if(e==null)return null;if(typeof e==`string`)return Gv(e);if(typeof e==`number`||typeof e==`boolean`)return e;if(e instanceof Array){let n=[],r=e.length;for(let i=0;i<r;++i){let r=e[i];CS(t,i,r)?n.push(r):n.push(TS(r,t))}return n}{let t={};for(let n of Object.keys(e)){let r=e[n],i=Gv(n);t[i]=(n===`name`||n===`className`)&&typeof r==`string`?r:TS(r,n)}return t}}var ES=`4.22.0`,DS=e=>{let t=Object.keys(e);if(t.length===0)return!1;let n=t[0].split(`/`);return!isNaN(parseInt(n[n.length-1],10))},OS=class e extends Wb{constructor(t){if(super({}),this.containerNodes=new Set,this.name=t.name,this.name==null){let e=this.getClassName().toLowerCase();this.name=uy(e)}if(this.supportsMasking=!1,this.trainable_=!0,this.inputs=Array.isArray(t.inputs)?t.inputs.slice():[t.inputs],this.outputs=Array.isArray(t.outputs)?t.outputs.slice():[t.outputs],$v(this.inputs).length!==this.inputs.length)throw new K(`The list of inputs passed to the model is redundant. All inputs should only appear once. Found: ${this.inputs.map(e=>e.name)}`);$v(this.outputs).length!==this.outputs.length&&console.warn(`The list of outputs passed to the model is redundant. All outputs should only appear once. Found: ${this.outputs.map(e=>e.name)}`),this.inputLayers=[],this.inputLayersNodeIndices=[],this.inputLayersTensorIndices=[],this.outputLayers=[],this.outputLayersNodeIndices=[],this.outputLayersTensorIndices=[],this.layers=[],this.internalContainerRefs=[];for(let e of this.outputs){let t=e.sourceLayer,n=e.nodeIndex,r=e.tensorIndex;this.outputLayers.push(t),this.outputLayersNodeIndices.push(n),this.outputLayersTensorIndices.push(r)}for(let e of this.inputs){let t=e.sourceLayer,n=e.nodeIndex,r=e.tensorIndex;Vv(n===0,`input layer has >1 nodes`),Vv(r===0,`input layer has >1 tensors`),this.inputLayers.push(t),this.inputLayersNodeIndices.push(n),this.inputLayersTensorIndices.push(r)}this.inputNames=[],this.outputNames=[],this.feedInputShapes=[],this.feedInputNames=[],this.feedOutputNames=[];for(let e=0;e<this.inputLayers.length;e++){let n=this.inputLayers[e];if(!(n instanceof Xb))throw TypeError(`Input layers to a LayersModel must be InputLayer objects. Received inputs: ${t.inputs}. Input ${e} (0-based) originates from layer type ${n.getClassName()}.`);this.inputNames.push(n.name),this.feedInputShapes.push(n.batchInputShape),this.feedInputNames.push(n.name)}for(let e of this.outputLayers)this.outputNames.push(e.name);this.internalInputShapes=this.inputs.map(e=>e.shape),this.internalOutputShapes=this.outputs.map(e=>e.shape);let n={},r={},i={},a={},o={},s=[],c=(t,n,r,i,a,l)=>{(i==null||a==null||l==null)&&(i=t.sourceLayer,a=t.nodeIndex,l=t.tensorIndex);let u=i.inboundNodes[a];if(r.indexOf(u)!==-1)throw new Iv(`The tensor ${t.name} at layer "${i.name}" is part of a cycle.`);if(n.indexOf(u)!==-1)return;this.containerNodes.add(e.nodeKey(i,a)),i.id in o||(o[i.id]=Object.keys(o).length),r.indexOf(u)===-1&&r.push(u);let d=u.inboundLayers.length;for(let e=0;e<d;e++){let t=u.inputTensors[e],i=u.inboundLayers[e],a=u.nodeIndices[e],o=u.tensorIndices[e];c(t,n,r,i,a,o)}for(n.push(u);r.indexOf(u)>=0;)r.splice(r.indexOf(u),1);s.push(u)},l=[],u=[];for(let e of this.outputs)c(e,l,u);let d=s.slice().reverse();for(let e of d){r[e.id]=e,e.id in n||(n[e.id]=0);let t=n[e.id],o=i[e.outboundLayer.id]==null?0:i[e.outboundLayer.id];t=Math.max(t,o),i[e.outboundLayer.id]=t,a[e.outboundLayer.id]=e.outboundLayer,n[e.id]=t;for(let i=0;i<e.inboundLayers.length;i++){let a=e.inboundLayers[i],o=e.nodeIndices[i],s=a.inboundNodes[o],c=n[s.id]==null?0:n[s.id];n[s.id]=Math.max(t+1,c),r[s.id]=s}}let f={};for(let e in n){let t=n[e];t in f||(f[t]=[]),f[t].push(r[e])}let p={};for(let e in i){let t=i[e];t in p||(p[t]=[]),p[t].push(a[e])}let m=Object.keys(p).map(e=>parseInt(e,10)).sort(Qv);this.layers=[];for(let t of m){let n=p[t];n.sort((e,t)=>{let n=o[e.id],r=o[t.id];return n<r?-1:+(n>r)});for(let t of n)t instanceof e&&this.internalContainerRefs.push(t),this.layers.push(t)}this.layersByDepth=p,m=Object.keys(f).map(e=>parseInt(e,10)).sort(Qv);let h=this.inputs.slice(),g=[];for(let e of m)for(let t of f[e]){let e=t.outboundLayer;if(e!=null){for(let n of t.inputTensors)if(h.indexOf(n)===-1)throw new Iv(`Graph disconnected: cannot obtain value for tensor ${n} at layer "${e.name}". The following previous layers were accessed without issue: ${g}`);for(let e of t.outputTensors)h.push(e);g.push(e.name)}}this.nodesByDepth=f;let _=this.layers.map(e=>e.name);for(let e of _){let t=_.filter(t=>t===e).length;if(t!==1)throw new Iv(`The name "${e}" is used ${t} times in the model. All layer names should be unique. Layer names: `+JSON.stringify(_))}this.outboundNodes=[],this.inboundNodes=[],new Hb({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:this.inputs.map(e=>null),outputMasks:this.outputs.map(e=>null),inputShapes:this.inputs.map(e=>e.shape),outputShapes:this.outputs.map(e=>e.shape)}),this.built=!0,this._refCount=1}assertNotDisposed(){if(this._refCount===0)throw Error(`Container '${this.name}' is already disposed.`)}dispose(){this.assertNotDisposed();let e={refCountAfterDispose:null,numDisposedVariables:0};if(--this._refCount===0){for(let t of this.layers)e.numDisposedVariables+=t.dispose().numDisposedVariables;for(let t of this.internalContainerRefs)e.numDisposedVariables+=t.dispose().numDisposedVariables}return e.refCountAfterDispose=this._refCount,e}get trainable(){return this.trainable_}set trainable(e){this.layers.forEach(t=>{t._trainableWeights.forEach(t=>t.trainable=e)}),this.trainable_=e}get trainableWeights(){if(this._trainableWeights.length>0)throw new K(`Container instance unexpectedly contains _trainableWeights.The trainable weights of a Container are a union of the trainable weights of its consituent Layers. Its own _trainableWeights must remain an empty Array.`);if(!this.trainable)return[];let e=[];for(let t of this.layers)e=e.concat(t.trainableWeights);return e}get nonTrainableWeights(){let e=[];for(let t of this.layers)e.push(...t.nonTrainableWeights);if(!this.trainable){let t=[];for(let e of this.layers)t.push(...e.trainableWeights);return t.concat(e)}return e}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}loadWeights(e,t=!0){let n={},r=0,i=DS(e);i&&this.parseWeights(e);for(let e of this.layers)for(let[t,a]of e.weights.entries()){let e=i?`${a.name.split(`/`).slice(0,-1).join(`/`)+`/`}${t}`:a.originalName;if(n[e]!=null)throw new K(`Duplicate weight name: ${e}`);n[e]=a,r++}let a=[];for(let r in e){let i=r;if(n[r]==null){let e=r.split(`/`);i=e.slice(0,-2).concat([e[e.length-1]]).join(`/`)}if(n[i]!=null)a.push([n[i],e[r]]);else if(t)throw new K(`Provided weight data has no target variable: ${r}`);delete n[i]}if(t){let e=[];for(let t in n)e.push(t);if(e.length>0)throw new K(`${e.length} of ${r} weights are not set: ${e}`)}Rb(a)}parseWeights(e){for(let t in Object.keys(e)){let n=t.split(`/`),r=[`vars`,`layer_checkpoint_dependencies`],i=n.map(e=>e.startsWith(`_`)?e.slice(1):e).filter(e=>!r.includes(e)).join(`/`);i!==t&&(e[i]=e[t],delete e[t])}}updatedConfig(){let e=this.getConfig(),t={};return t.className=this.getClassName(),t.config=e,t.kerasVersion=`tfjs-layers ${ES}`,t.backend=`TensorFlow.js`,t}toJSON(e,t=!0){let n=TS(this.updatedConfig());return t?JSON.stringify(n):n}call(e,t){return F(()=>{e=Wv(e);let n=new $b;for(let t=0;t<this.inputs.length;++t)n.add(this.inputs[t],e[t]);return rx(this.outputs,n,t)})}computeMask(e,t){return F(()=>{e=Wv(e);let n;return n=t==null?Bv(null,e.length):Wv(t),this.runInternalGraph(e,n)[1]})}computeOutputShape(e){let t=jb(e);if(t.length!==this.inputLayers.length)throw new K(`Invalid inputShape argument ${e}: model has ${this.inputLayers.length} tensor inputs.`);let n={};for(let e=0;e<t.length;e++){let r=this.inputLayers[e],i=t[e],a=r.name+`_0_0`;n[a]=i}let r=Object.keys(this.nodesByDepth).map(e=>parseInt(e,10)).sort(Qv);if(r.length>1)for(let e of r){let t=this.nodesByDepth[e];for(let e of t){let t=e.outboundLayer;if(this.inputLayers.map(e=>e.id).indexOf(t.id)!==-1)continue;let r=[];for(let t=0;t<e.inboundLayers.length;t++){let i=e.inboundLayers[t],a=e.nodeIndices[t],o=e.tensorIndices[t],s=n[`${i.name}_${a}_${o}`];r.push(s)}let i=jb(t.computeOutputShape(Uv(r))),a=t.inboundNodes.indexOf(e);for(let e=0;e<i.length;e++){let r=`${t.name}_${a}_${e}`;n[r]=i[e]}}}let i=[],a=[];for(let e=0;e<this.outputLayers.length;e++){let t=this.outputLayers[e],n=this.outputLayersNodeIndices[e],r=this.outputLayersTensorIndices[e],i=`${t.name}_${n}_${r}`;a.push(i)}for(let e=0;e<a.length;e++){let t=a[e];Vv(t in n),i.push(n[t])}return Uv(i)}runInternalGraph(e,t){t??=Bv(null,e.length);let n={};for(let r=0;r<this.inputs.length;++r){let i=this.inputs[r],a=e[r],o=t[r];n[i.id]=[a,o]}let r=Object.keys(this.nodesByDepth).map(e=>parseInt(e,10)).sort(Qv);for(let e of r){let t=this.nodesByDepth[e];for(let e of t){let t=e.outboundLayer,r=e.inputTensors,i=e.outputTensors,a=[];for(let e of r)e.id in n&&a.push(n[e.id]);if(a.length===r.length){let r={},o,s,c,l;if(e.callArgs!=null&&(r=e.callArgs),a.length===1){let[e,n]=a[0];r.mask??(r.mask=n),c=Wv(t.call(e,r)),l=Wv(t.computeMask(e,n)),o=[e],s=[n]}else o=a.map(e=>e[0]),s=a.map(e=>e[1]),r.mask??(r.mask=s),c=Wv(t.call(o,r)),l=Wv(t.computeMask(o,s));if(t.activityRegularizer)throw new Lv(`LayersModel invocation with concrete Tensor value(s) in the presence of activity regularizer(s) is not supported yet.`);for(let e=0;e<i.length;++e){let t=i[e],r=c[e],a=l[e];n[t.id]=[r,a]}}}}let i=[],a=[],o=[];for(let e of this.outputs){Vv(e.id in n,`Could not compute output ${e.name} : ${e.id}`);let[t,r]=n[e.id];o.push(t.shape),i.push(t),a.push(r)}return[i,a,o]}buildNodeConversionMap(t){let n={},r;for(let t of this.layers){r=+(t instanceof e);for(let i=0;i<t.inboundNodes.length;i++){let a=e.nodeKey(t,i);this.containerNodes.has(a)&&(n[a]=r,r+=1)}}return n}getLayer(e,t){if(t!=null)return this.findLayer(t);if(e==null)throw new K(`Provide either a layer name or layer index`);if(typeof e==`number`)return this.findLayer(e);for(let t of this.layers)if(t.name===e)return t;throw new K(`No such layer: ${e}`)}findLayer(e){if(this.layers.length<=e)throw new K(`Was asked to retrieve layer at index ${e}, but model only has ${this.layers.length} layer(s).`);return this.layers[e]}calculateLosses(){return F(()=>{let t=[];for(let n of this.layers)for(let r=0;r<n.inboundNodes.length;++r){let i=e.nodeKey(n,r);this.containerNodes.has(i)&&t.push(...n.calculateLosses())}return t})}getConfig(){let t={name:this.name},n=this.buildNodeConversionMap(this.layers),r=[];for(let t of this.layers){let i=t.getClassName(),a=t.getConfig(),o=[];for(let r=0;r<t.inboundNodes.length;r++){let i=t.inboundNodes[r],a=e.nodeKey(t,r),s={};if(this.containerNodes.has(a)){if(i.callArgs)try{JSON.stringify(i.callArgs),s=i.callArgs}catch{console.warn(`Layer ${t.name} was passed non-serializable keyword arguments: ${i.callArgs}. They will not be included in the serialized model (and thus will be missing at deserialization time).`),s={}}if(i.inboundLayers.length>0){let t=[];for(let r=0;r<i.inboundLayers.length;r++){let a=i.inboundLayers[r],o=i.nodeIndices[r],c=i.tensorIndices[r],l=n[e.nodeKey(a,o)];l??=0,t.push([a.name,l,c,s])}o.push(t)}}}let s={};s.name=t.name,s.className=i,s.config=a,s.inboundNodes=o,r.push(s)}t.layers=r;let i=[];for(let t=0;t<this.inputLayers.length;t++){let r=this.inputLayers[t],a=this.inputLayersNodeIndices[t],o=e.nodeKey(r,a);if(!this.containerNodes.has(o))continue;let s=n[o];s??=0;let c=this.inputLayersTensorIndices[t];i.push([r.name,s,c])}t.inputLayers=i;let a=[];for(let t=0;t<this.outputLayers.length;t++){let r=this.outputLayers[t],i=this.outputLayersNodeIndices[t],o=e.nodeKey(r,i);if(!this.containerNodes.has(o))continue;let s=n[o];s??=0;let c=this.outputLayersTensorIndices[t];a.push([r.name,s,c])}return t.outputLayers=a,t}static fromConfig(e,t,n={},r=!1){let i={},a={};function o(e,t){e.name in a?a[e.name].push(t):a[e.name]=[t]}function s(e,t){let n=[],r;for(let a of t){let s=a[0],c=a[1],l=a[2];if(r=a[3]==null?{}:a[3],!(s in i)){o(e,t);return}let u=i[s];if(u.inboundNodes.length<=c){o(e,t);return}let d=u.inboundNodes[c];n.push(d.outputTensors[l])}n.length>0&&e.apply(Uv(n),r)}function c(e){let n=e.name,a=kx(e,t.customObjects==null?{}:t.customObjects);a.setFastWeightInitDuringBuild(r),i[n]=a,e.inboundNodes.forEach(e=>{if(!(e instanceof Array))throw new K(`Corrupted configuration, expected array for nodeData: ${e}`);o(a,e)})}let l=t.name,u=t.layers;for(let e of u)c(e);for(;!ey(a);)for(let e of u){let t=i[e.name];if(t.name in a){let e=a[t.name];delete a[t.name];for(let n of e)s(t,n)}}let d=[],f=[],p=t.inputLayers;for(let e of p){let t=e[0],n=e[1],r=e[2];Vv(t in i);let a=i[t].inboundNodes[n].outputTensors;d.push(a[r])}let m=t.outputLayers;for(let e of m){let t=e[0],n=e[1],r=e[2];Vv(t in i);let a=i[t].inboundNodes[n].outputTensors;f.push(a[r])}return new e({inputs:d,outputs:f,name:l})}get stateful(){if(this._stateful)throw new K(`Container instance unexpectedly has _stateful = true. The statefulness of a Container is determined by the Layers it contains. Its _stateful property must remain the default false.`);for(let e of this.layers)if(e.stateful)return!0;return!1}resetStates(){F(()=>{this.layers.forEach(e=>{e.stateful&&e.resetStates()})})}};function kS(e,t,n){let r=t.length;if(e==null||Array.isArray(e)&&e.length===0)return t.map(e=>null);if(r===1)return Array.isArray(e)&&e.length===1?e:typeof e==`object`&&t[0]in e?[e[t[0]]]:[e];if(Array.isArray(e)){if(e.length!==r)throw Error(`Provided ${n} is an array of ${e.length} element(s), but the model has ${r} outputs. Make sure a set of weights is provided for each model output.`);return e}if(typeof e==`object`&&Object.keys(e).length>0&&typeof e[Object.keys(e)[0]]==`object`){let n=[];return t.forEach(t=>{t in e?n.push(e[t]):n.push(null)}),n}throw Error(`The model has multiple (${r}) outputs, so ${n} must be either an array with ${r} elements or an object with ${t} keys. Provided ${n} not understood: ${JSON.stringify(e)}`)}function AS(e,t){return kS(e,t,`classWeight`)}async function jS(e,t,n,r){if(t!=null||r!=null)throw Error(`Support sampleWeight is not implemented yet`);if(n!=null){let t=F(()=>{if(e.shape.length===1)return bo(e);if(e.shape.length===2){if(e.shape[1]>1)return Bo(e,1);if(e.shape[1]===1)return B(e,[e.shape[0]]);throw Error(`Encountered unexpected last-dimension size (${e.shape[1]}) during handling of class weights. The size is expected to be >= 1.`)}throw Error(`Unexpected rank of target (y) tensor (${e.rank}) during handling of class weights. The rank is expected to be 1 or 2.`)}),r=Array.from(await t.data());ya(t);let i=[];return r.forEach(e=>{if(n[e]==null)throw Error(`classWeight must contain all classes in the training data. The class ${e} exists in the data but not in classWeight`);i.push(n[e])}),Af(i,`float32`)}return null}function MS(e,t){return z(e,t)}var NS=32;function PS(e,t){let n,r,i=t;n=i.xs,r=i.ys,b(n!=null&&r!=null,()=>`A Dataset iterator for fitDataset() is expected to generate objects of the form \`{xs: xVal, ys: yVal}\`, where the two values may be \`tf.Tensor\`, an array of Tensors, or a map of string to Tensor.  The provided Dataset instead generates ${t}`);let a=FS(`input`,e.inputNames,n),o=FS(`output`,e.outputNames,r),s=a[0].shape[0];b(a.length===e.inputs.length,()=>`LayersModel has ${e.inputs.length} inputs, but the dataset provides ${a.length} inputs.  (Expected input keys: ${JSON.stringify(e.inputNames)})`),b(o.length===e.outputs.length,()=>`LayersModel has ${e.outputs.length} outputs, but the dataset provides ${o.length} outputs.  (Expected output keys: ${JSON.stringify(e.outputNames)})`);for(let t=0;t<a.length;t++)b(a[t].shape[0]===s,()=>`Batch size mismatch: input ${e.inputNames[t]} has ${a[t].shape[0]}; expected  ${s} based on input ${e.inputNames[0]}.`);for(let t=0;t<o.length;t++)b(o[t].shape[0]===s,()=>`Batch size mismatch: output ${e.outputNames[t]} has ${o[t].shape[0]}; expected  ${s} based on input ${e.inputNames[0]}.`);return{xs:a,ys:o}}function FS(e,t,n){if(n instanceof Ni)return[n];if(Array.isArray(n))return b(n.length===t.length,()=>`Received an array of ${n.length} Tensors, but expected ${t.length} to match the ${e} keys ${t}.`),n;{let r=[];for(let i of t){if(n[i]==null)throw new K(`The feature data generated by the dataset lacks the required ${e} key '${i}'.`);r.push(n[i])}return r}}function IS(e){if(e.length===3)throw new Lv(`Validation with sample weights is not implemented yet.`);return{xs:e[0],ys:e[1]}}async function LS(e,t,n){let r=n.batchesPerEpoch!=null;if(b(e.optimizer!=null,()=>`You must compile a model before training/testing. Use LayersModel.compile(modelCompileConfig).`),b(n!=null,()=>`For fitDataset(), the 2nd argument (config) is required, but it is not provided in this call.`),b(n.epochs!=null&&n.epochs>0&&Number.isInteger(n.epochs),()=>`For fitDataset(), config.epochs is expected to be a positive integer, but got ${n.epochs}`),b(!r||n.batchesPerEpoch>0&&Number.isInteger(n.batchesPerEpoch),()=>`For fitDataset(), config.batchesPerEpoch is expected to be a positive integer if specified, but got ${n.batchesPerEpoch}`),b(n.validationSplit==null,()=>"`validationSplit` is not supported by `fitDataset()`. Use validationData instead."),e.isTraining)throw Error(`Cannot start training because another fit() call is ongoing.`);e.isTraining=!0;try{let i=n.validationData!=null,a,o;if(i){if(zS(n.validationData))b(n.validationBatches==null||n.validationBatches>0&&Number.isInteger(n.validationBatches),()=>`For fitDataset() with dataset-based validation, config.validationBatches is expected not to be provided, or to be a positive integer, but got ${n.validationBatches}`);else{let e=IS(n.validationData);a=e.xs,o=e.ys}}let s=e.makeTrainFunction(),c=e.getDedupedMetricsNames(),l;l=i?c.slice().concat(c.map(e=>`val_`+e)):c.slice();let{callbackList:u,history:d}=Ox(Ex(n.callbacks,n.yieldEvery),n.verbose==null?1:n.verbose,n.epochs,null,null,RS(t,n),null,i,l);u.setModel(e),e.history=d,await u.onTrainBegin(),e.stopTraining_=!1;let f=n.initialEpoch==null?0:n.initialEpoch,p=await t.iterator();for(;f<n.epochs;){let l={};await u.onEpochBegin(f);let d=0,m=0;for(r||(p=await t.iterator());!r||d<n.batchesPerEpoch;){let t=await p.next();if(r&&t.done){console.warn(`You provided \`batchesPerEpoch\` as ${n.batchesPerEpoch}, but your dataset iterator ran out of data after ${d} batches; interrupting training. Make sure that your dataset can generate at least \`batchesPerEpoch * epochs\` batches (in this case, ${n.batchesPerEpoch*n.epochs} batches). You may need to use the repeat() function when building your dataset.`);break}if(t.value!=null){let{xs:r,ys:i}=PS(e,t.value),a={};a.batch=m,a.size=r[0].shape[0],await u.onBatchBegin(m,a);let o=[];if(n.classWeight!=null){let t=AS(n.classWeight,e.outputNames);for(let e=0;e<t.length;++e)o.push(await jS(i[e],null,t[e]))}let l=r.concat(i).concat(o),f=s(l);ya(l);for(let e=0;e<c.length;++e){let t=c[e],n=f[e];a[t]=n,ba(n)}await u.onBatchEnd(m,a),yx(a),m++,d++}if(r?d>=n.batchesPerEpoch:t.done){if(i){let t;t=zS(n.validationData)?Wv(await e.evaluateDataset(n.validationData,{batches:n.validationBatches})):Wv(e.evaluate(a,o,{batchSize:n.validationBatchSize==null?NS:n.validationBatchSize,verbose:0}));for(let n=0;n<e.metricsNames.length;++n)l[`val_${e.metricsNames[n]}`]=t[n]}break}if(e.stopTraining_)break}if(await u.onEpochEnd(f,l),f++,e.stopTraining_)break}return await u.onTrainEnd(),await e.history.syncData(),e.history}finally{e.isTraining=!1}}function RS(e,t){let n=null;return t.batchesPerEpoch==null?Number.isFinite(e.size)&&(n=e.size):n=t.batchesPerEpoch,n}function zS(e){return typeof e.iterator==`function`}function BS(e){return typeof e.next==`function`}async function VS(e,t,n){n||={};let r=n.batches!=null,i=e.testFunction,a=[];if(n.verbose>0)throw new Lv(`Verbose mode is not implemented yet.`);b(!r||n.batches>0&&Number.isInteger(n.batches),()=>`Test loop expects \`batches\` to be a positive integer, but received ${JSON.stringify(n.batches)}`);let o=BS(t)?t:await t.iterator(),s=0,c=0;for(;!r||c<n.batches;){let t=await o.next();if(a=F(()=>{if(t.value){let{xs:n,ys:r}=PS(e,t.value),o=n.concat(r),l=F(()=>i(o));if(ya(o),c===0)for(let e=0;e<l.length;++e)a.push(H(0));let u=o[0].shape[0];for(let e=0;e<l.length;++e){let t=l[e],n=a[e];a[e]=F(()=>L(a[e],z(u,t))),c>0&&ya(n)}ya(l),s+=u,++c}return a}),t.done){r&&console.warn(`Your dataset iterator ran out of data during evaluateDataset(). Interrupting evalution. Make sure that your dataset can generate at least \`batches\` batches (in this case, ${n.batches} batches). You may need to use the repeat() function when building your dataset.`);break}}for(let e=0;e<a.length;++e){let t=a[e];a[e]=R(a[e],s),ya(t)}return Uv(a)}function HS(e){b(e>0&&Number.isInteger(e),()=>`batchSize is required to be a positive integer, but got ${e}`)}function US(e,t,n){return e==null?[null]:Array.isArray(e)?e.map(e=>Hy(e,t,n-t)):Hy(e,t,n-t)}function WS(e,t){return F(()=>e==null?null:Array.isArray(e)?e.map(e=>WS(e,t)):Xy(e,t.dtype===`int32`?t:I(t,`int32`)))}function GS(e,t){let n=[],r=0,i=null;for(;r<e;)i=r+t,i>=e&&(i=e),n.push([r,i]),r=i;return n}function KS(e){let t=[];e instanceof Ni&&(e=[e]);for(let n=0;n<e.length;++n){let r=e[n];if(r.rank===1)t.push(Ry(r,1));else if(r.rank===0)throw Error(`Expected tensor to be at least 1D, but received a 0D tensor (scalar).`);else t.push(r)}return t}function qS(e,t){if(e==null)return;let n=[];if(t instanceof Ni)n.push(t.id);else if(Array.isArray(t))t.forEach(e=>n.push(e.id));else if(t!=null)for(let e in t){let r=t[e];n.push(r.id)}let r=[];if(e instanceof Ni)n.indexOf(e.id)===-1&&r.push(e);else if(Array.isArray(e))e.forEach(e=>{n.indexOf(e.id)===-1&&r.push(e)});else if(e!=null)for(let t in e){let i=e[t];n.indexOf(i.id)===-1&&r.push(i)}r.forEach(e=>{e.isDisposed||e.dispose()})}function JS(e){return e instanceof Ni}function YS(e){return Array.isArray(e)}function XS(e){return!JS(e)&&!YS(e)}function ZS(e,t,n,r=!0,i=``){if(t==null||t.length===0){if(e!=null){let t=!1;if(YS(e)&&e.length>0)t=!0;else if(XS(e)){for(let n in e)if(e.hasOwnProperty(n)){t=!0;break}}else t=!0;if(t)throw new K(`Error when checking model ${i} expected no data, but got ${e}`)}return[]}if(e==null)return t.map(e=>null);let a;if(XS(e)){e=e,a=[];for(let n of t){if(e[n]==null)throw new K(`No data provided for "${n}". Need data for each key in: ${t}`);a.push(e[n])}}else if(YS(e)){if(e=e,e.length!==t.length)throw new K(`Error when checking model ${i}: the Array of Tensors that you are passing to your model is not the size the model expected. Expected to see ${t.length} Tensor(s), but instead got the following list of Tensor(s): ${e}`);a=e}else{if(e=e,t.length>1)throw new K(`The model ${i} expects ${t.length} Tensor(s), but only received one Tensor. Found: Tensor with shape ${e.shape}`);a=[e]}if(a=KS(a),n!=null)for(let e=0;e<t.length;++e){if(n[e]==null)continue;let o=a[e];if(o.shape.length!==n[e].length)throw new K(`Error when checking ${i}: expected ${t[e]} to have ${n[e].length} dimension(s). but got array with shape ${o.shape}`);for(let t=0;t<n[e].length;++t){if(t===0&&!r)continue;let a=o.shape[t],s=n[e][t];if(s!=null&&s>=0&&a!==s)throw new K(`${i} expected a batch of elements where each example has shape [${n[e].slice(1,n[e].length)}] (i.e.,tensor shape [*,${n[e].slice(1,n[e].length)}]) but the ${i} received an input with ${o.shape[0]} examples, each with shape [${o.shape.slice(1,o.shape.length)}] (tensor shape [${o.shape}])`)}}return a}function QS(e,t,n){let r=$v(e.map(e=>e.shape[0]));r.sort();let i=$v(t.map(e=>e.shape[0]));if(i.sort(),r.length>1)throw new K(`All input Tensors (x) should have the same number of samples. Got array shapes: ${JSON.stringify(e.map(e=>e.shape))}`);if(i.length>1)throw new K(`All target Tensors (y) should have the same number of samples. Got array shapes: ${JSON.stringify(t.map(e=>e.shape))}`);if(r.length>0&&i.length>0&&!w(r,i))throw new K(`Input Tensors should have the same number of samples as target Tensors. Found ${r[0]} input sample(s) and ${i[0]} target sample(s).`)}function $S(e,t,n){let r=[jx,Hx,zx];for(let i=0;i<e.length;++i){let a=e[i],o=t[i],s=n[i];if(o!=null){if(o===zx&&a.shape[a.shape.length-1]===1)throw new K(`You are passing a target array of shape ${a.shape} while using a loss 'categorical_crossentropy'. 'categorical_crossentropy'expects targets to be binary matrices (1s and 0s) of shape [samples, classes].`);if(r.indexOf(o)!==-1){let e=a.shape.slice(1),t=s.slice(1);for(let n=0;n<e.length;++n){let r=e[n],i=t[n];if(i!=null&&r!==i)throw new K(`A target Tensor with shape ${a.shape} was passed for an output of shape ${s}, while using a loss function that expects targets to have the same shape as the output.`)}}}}}function eC(e,t,n,r=!0,i=``){let a;if(Array.isArray(e)){if(e.length!==t.length)throw new K(`Error when checking model ${i}: the Array of Tensors that you are passing to your model is not the size the the model expected. Expected to see ${t.length} Tensor(s), but instead got ${e.length} Tensors(s).`);a=e}else{if(t.length>1)throw new K(`The model expects ${t.length} ${i} Tensors, but only received one Tensor. Found: array with shape ${JSON.stringify(e.shape)}.`);a=[e]}if(n!=null)for(let e=0;e<t.length;++e){if(n[e]==null)continue;let o=a[e];if(o.shape.length!==n[e].length)throw new K(`Error when checking ${i}: expected ${t[e]} to have ${n[e].length} dimension(s), but got array with shape ${JSON.stringify(o.shape)}`);for(let a=0;a<n[e].length;++a){if(a===0&&!r)continue;let s=o.shape[a],c=n[e][a];if(c!=null&&c!==s)throw new K(`Error when checking ${i}: expected ${t[e]} to have shape ${JSON.stringify(n[e])} but got array with shape ${JSON.stringify(o.shape)}.`)}}}function tC(e,t){if(e==null||Array.isArray(e)&&e.length===0)return t.map(e=>[]);let n;if(typeof e==`string`||typeof e==`function`)n=[e];else if(Array.isArray(e)||typeof e==`object`)n=e;else throw TypeError(`Type of metrics argument not understood. Expected an string,function, Array, or Object, found: ${e}`);if(Array.isArray(n))return t.map(e=>n);{let e=[];for(let r of t){let t=n.hasOwnProperty(r)?n[r]:[];Array.isArray(t)||(t=[t]),e.push(t)}return e}}var nC=`layers-model`,rC=class extends OS{constructor(e){super(e),this.isTraining=!1}summary(e,t,n=console.log){if(!this.built)throw new K(`This model has never been called, thus its weights have not been created yet. So no summary can be displayed. Build the model first (e.g., by calling it on some test data).`);_S(this,e,t,n)}compile(e){if(e.loss??=[],this.loss=e.loss,typeof e.optimizer==`string`)this.optimizer_=pS(e.optimizer),this.isOptimizerOwned=!0;else{if(!(e.optimizer instanceof Lm))throw new K(`User-defined optimizer must be an instance of tf.Optimizer.`);this.optimizer_=e.optimizer,this.isOptimizerOwned=!1}let t=[];if(!Array.isArray(e.loss)&&typeof e.loss!=`string`&&typeof e.loss!=`function`){e.loss=e.loss;for(let t in e.loss)if(this.outputNames.indexOf(t)===-1)throw new K(`Unknown entry in loss dictionary: "${t}". Only expected the following keys: ${this.outputNames}`);for(let n of this.outputNames)e.loss[n]??console.warn(`Output "${n}" is missing from loss dictionary. We assume this was done on purpose, and we will not be expecting data to be passed to ${n} during training`),t.push(qx(e.loss[n]))}else if(Array.isArray(e.loss)){if(e.loss.length!==this.outputs.length)throw new K(`When passing an Array as loss, it should have one entry per model output. The model has ${this.outputs.length} output(s), but you passed loss=${e.loss}.`);t=e.loss.map(e=>qx(e))}else{let n=qx(e.loss);this.outputs.forEach(e=>{t.push(n)})}this.lossFunctions=t,this.feedOutputNames=[],this.feedOutputShapes=[],this.feedLossFns=[];for(let e=0;e<this.outputs.length;++e){let t=this.internalOutputShapes[e],n=this.outputNames[e];this.feedOutputNames.push(n),this.feedOutputShapes.push(t),this.feedLossFns.push(this.lossFunctions[e])}let n=[];this.metrics=e.metrics,this.metricsNames=[`loss`],this.metricsTensors=[],Cy(`loss`,()=>{for(let e=0;e<this.outputs.length;++e){if(n.indexOf(e)!==-1)continue;let t=this.lossFunctions[e];this.outputs.length>1&&(this.metricsTensors.push([t,e]),this.metricsNames.push(this.outputNames[e]+`_loss`))}});let r=tC(e.metrics,this.outputNames),i=(e,t,n)=>{this.outputNames.length>1&&(t=this.outputNames[e]+`_`+t),this.metricsNames.push(t),this.metricsTensors.push([n,e])};Cy(`metric`,()=>{for(let e=0;e<this.outputs.length;++e)n.indexOf(e)===-1&&(t=>{let n,r,a;for(let o of t){if(typeof o==`string`&&[`accuracy`,`acc`,`crossentropy`,`ce`].indexOf(o)!==-1){let t=this.internalOutputShapes[e];t[t.length-1]===1||this.lossFunctions[e]===Hx?[`accuracy`,`acc`].indexOf(o)===-1?[`crossentropy`,`ce`].indexOf(o)!==-1&&(r=$x):r=Jx:this.lossFunctions[e]===Bx?[`accuracy`,`acc`].indexOf(o)===-1?[`crossentropy`,`ce`].indexOf(o)!==-1&&(r=lS):r=eS:[`accuracy`,`acc`].indexOf(o)===-1?[`crossentropy`,`ce`].indexOf(o)!==-1&&(r=sS):r=Yx;let i;[`accuracy`,`acc`].indexOf(o)===-1?[`crossentropy`,`ce`].indexOf(o)!==-1&&(i=`ce`):i=`acc`,a=r,n=``+i}else a=dS(o),n=``+fS(o);let t;Cy(n,()=>{t=a}),i(e,n,t)}})(r[e])}),this.collectedTrainableWeights=this.trainableWeights}checkTrainableWeightsConsistency(){this.collectedTrainableWeights!=null&&this.trainableWeights.length!==this.collectedTrainableWeights.length&&console.warn("Discrepancy between trainableweights and collected trainable weights. Did you set `model.trainable` without calling `model.compile()` afterwards?")}evaluate(e,t,n={}){let r=n.batchSize==null?32:n.batchSize;HS(r);let i=this.standardizeUserDataXY(e,t,!0,r);try{let e=i[0].concat(i[1]);this.makeTestFunction();let t=this.testFunction;return Uv(this.testLoop(t,e,r,n.verbose,n.steps))}finally{qS(i[0],e),qS(i[1],t)}}async evaluateDataset(e,t){return this.makeTestFunction(),VS(this,e,t)}checkNumSamples(e,t,n,r=`steps`){let i;if(n!=null){if(i=null,t!=null)throw new K(`If ${r} is set, batchSize must be null or undefined.Got batchSize = ${t}`)}else if(e!=null)i=Array.isArray(e)?e[0].shape[0]:e.shape[0];else throw new K(`Either the input data should have a defined shape, or ${r} shoud be specified.`);return i}execute(e,t){if(Array.isArray(t)&&t.length===0)throw new K("`outputs` is an empty Array, which is not allowed.");let n=Array.isArray(t),r=n?t:[t],i=this.retrieveSymbolicTensors(r),a=new $b;if(e instanceof Ni&&(e=[e]),Array.isArray(e)){if(e.length!==this.inputs.length)throw new K(`The number of inputs provided (${e.length}) does not match the number of inputs of this model (${this.inputs.length}).`);for(let t=0;t<this.inputs.length;++t)a.add(this.inputs[t],e[t])}else for(let t of this.inputs){let n=e[t.name];if(n==null)throw new K(`No value is provided for the model's input ${t.name}`);a.add(t,n)}let o=rx(i,a);return n?o:o[0]}retrieveSymbolicTensors(e){let t=Bv(null,e.length),n=e.length;for(let r of this.layers){let i=Array.isArray(r.output)?r.output:[r.output],a=i.map(e=>e.name);for(let r=0;r<e.length;++r){let o=a.indexOf(e[r]);if(o!==-1&&(t[r]=i[o],n--),n===0)break}if(n===0)break}if(n>0){let n=[];throw t.forEach((t,r)=>{t??n.push(e[r])}),new K(`Cannot find SymbolicTensors for output name(s): ${JSON.stringify(n)}`)}return t}predictLoop(e,t=32,n=!1){return F(()=>{let r=this.checkNumSamples(e);if(n)throw new Lv(`Verbose predictLoop() is not implemented yet.`);let i=GS(r,t),a=this.outputs.map(e=>[]);for(let t=0;t<i.length;++t)F(()=>{let n=i[t][0],r=i[t][1],a=US(e,n,r),o=[];if(Array.isArray(a))for(let e=0;e<a.length;++e)o.push({key:this.inputs[e],value:a[e]});else o.push({key:this.inputs[0],value:a});let s=new $b(o);return rx(this.outputs,s)}).forEach((e,t)=>a[t].push(e));return Uv(a.map(e=>ws(e,0)))})}predict(e,t={}){let n=KS(e);eC(n,this.inputNames,this.feedInputShapes,!1);try{let e=t.batchSize==null?32:t.batchSize;return HS(e),this.predictLoop(n,e)}finally{qS(n,e)}}predictOnBatch(e){eC(e,this.inputNames,this.feedInputShapes,!0);let t=(Array.isArray(e)?e[0]:e).shape[0];return this.predictLoop(e,t)}standardizeUserDataXY(e,t,n=!0,r){if(this.optimizer_==null)throw new Iv(`You must compile a model before training/testing. Use LayersModel.compile(modelCompileArgs).`);let i=[];for(let e=0;e<this.feedOutputShapes.length;++e){let t=this.feedOutputShapes[e];this.feedLossFns[e]===Bx?i.push(t.slice(0,t.length-1).concat([1])):i.push(t)}if(e=ZS(e,this.feedInputNames,this.feedInputShapes,!1,`input`),t=ZS(t,this.feedOutputNames,i,!1,`target`),QS(e,t,null),$S(t,this.feedLossFns,this.feedOutputShapes),this.stateful&&r!=null&&r>0&&e[0].shape[0]%r!==0)throw new K(`In a stateful network, you should only pass inputs with a number of samples that is divisible by the batch size ${r}. Found: ${e[0].shape[0]} sample(s).`);return[e,t]}async standardizeUserData(e,t,n,r,i=!0,a){let[o,s]=this.standardizeUserDataXY(e,t,i,a);if(n!=null)throw Error(`sample weight is not supported yet.`);let c=null;if(r!=null){let e=AS(r,this.outputNames);c=[];for(let t=0;t<e.length;++t)c.push(await jS(s[t],null,e[t]))}return[o,s,c]}testLoop(e,t,n,r=0,i){return F(()=>{let a=this.checkNumSamples(t,n,i,`steps`),o=[];if(r>0)throw new Lv(`Verbose mode is not implemented yet.`);if(i!=null)throw new Lv(`steps mode in testLoop() is not implemented yet`);{let r=GS(a,n),i=Af(Ny(0,a));for(let n=0;n<r.length;++n){let a=r[n][0],s=r[n][1],c=e(WS(t,Hy(i,a,s-a)));if(n===0)for(let e=0;e<c.length;++e)o.push(H(0));for(let e=0;e<c.length;++e){let t=c[e];o[e]=L(o[e],z(s-a,t))}}for(let e=0;e<o.length;++e)o[e]=R(o[e],a)}return o})}getDedupedMetricsNames(){let e=this.metricsNames,t=[];for(let n=0;n<e.length;++n){let r=e[n],i=r;if(Hv(e,r)>1){let t=Hv(e.slice(0,n),r);i+=`_${t}`}t.push(i)}return t}makeTrainFunction(){return e=>{let t=[],n=e.slice(0,this.inputs.length),r=e.slice(this.inputs.length,this.inputs.length+this.outputs.length),i=e.slice(this.inputs.length+this.outputs.length,this.inputs.length+this.outputs.length*2),a=[],o=()=>{let e=[];for(let t=0;t<this.inputs.length;++t)e.push({key:this.inputs[t],value:n[t]});let o=new $b(e),s=rx(this.outputs,o,{training:!0}),c;for(let e=0;e<this.lossFunctions.length;++e){let n=this.lossFunctions[e],a=n(r[e],s[e]);i[e]!=null&&(a=MS(a,i[e]));let o=Lu(a);t.push(o),c=e===0?a:L(c,a)}for(let e=0;e<this.metricsTensors.length;++e){let n;if(this.outputs.length>1&&e<this.outputs.length)n=t[e];else{let t=this.metricsTensors[e][0],i=this.metricsTensors[e][1];n=Lu(t(r[i],s[i]))}ba(n),a.push(n)}return c=Lu(c),this.calculateLosses().forEach(e=>{c=L(c,e)}),c},s=this.collectedTrainableWeights.map(e=>e.read());return[this.optimizer_.minimize(o,!0,s)].concat(a)}}makeTestFunction(){this.testFunction=e=>F(()=>{let t=[],n,r=e.slice(0,this.inputs.length),i=e.slice(this.inputs.length,this.inputs.length+this.outputs.length),a=[];for(let e=0;e<this.inputs.length;++e)a.push({key:this.inputs[e],value:r[e]});let o=new $b(a),s=rx(this.outputs,o);for(let e=0;e<this.lossFunctions.length;++e){let r=this.lossFunctions[e],a=Lu(r(i[e],s[e]));n=e===0?a:L(n,a),t.push(n)}for(let e=0;e<this.metricsTensors.length;++e){let n=this.metricsTensors[e][0],r=this.metricsTensors[e][1],a=Lu(n(i[r],s[r]));t.push(a)}return t})}async fit(e,t,n={}){if(this.isTraining)throw Error(`Cannot start training because another fit() call is ongoing.`);this.isTraining=!0;let r,i,a,o,s,c,l,u,d;try{let f=n.batchSize==null?32:n.batchSize;HS(f);let p=await this.standardizeUserData(e,t,n.sampleWeight,n.classWeight,!1,f);r=p[0],i=p[1],d=p[2];let m=!1,h;if(n.validationData!=null&&n.validationData.length>0){if(m=!0,n.validationData.length===2)s=n.validationData[0],c=n.validationData[1];else if(n.validationData.length===3)throw new Lv(`validationData including sample weights is not supported yet.`);else throw new K(`When passing validation data, it must contain 2 (valX, valY) or 3 (valX, valY, valSampleWeight) items; ${n.validationData} is invalid.`);let e=await this.standardizeUserData(s,c,null,null,!0,f);l=e[0],u=e[1],h=l.concat(u)}else if(n.validationSplit!=null&&n.validationSplit>0&&n.validationSplit<1){m=!0;let e=Math.floor(r[0].shape[0]*(1-n.validationSplit)),t=r[0].shape[0];l=US(r,e,t),a=r,r=US(r,0,e),u=US(i,e,t),o=i,i=US(i,0,e),h=l.concat(u)}else n.validationSteps!=null&&(m=!0);let g=r.concat(i).concat(d);this.checkTrainableWeightsConsistency();let _=this.makeTrainFunction(),v=this.getDedupedMetricsNames(),y,b;m?(this.makeTestFunction(),y=this.testFunction,b=v.slice().concat(v.map(e=>`val_`+e))):(y=null,h=[],b=v.slice());let x=Ex(n.callbacks,n.yieldEvery);return await this.fitLoop(_,g,v,f,n.epochs,n.verbose,x,y,h,n.shuffle,b,n.initialEpoch,null,null)}finally{this.isTraining=!1,qS(r,e),qS(i,t),qS(a,e),qS(o,t),qS(l,s),qS(u,c),d!=null&&ya(d)}}async fitLoop(e,t,n,r,i,a,o,s,c,l,u,d,f,p){r??=32,i??=1,l??=!0,d??=0;let m=!1;if(s!=null&&c!=null&&(m=!0),p!=null&&(m=!0,f==null))throw new K("Can only use `validationSteps` when doing step-wise training, i.e., `stepsPerEpoch` must be set.");let g=this.checkNumSamples(t,r,f,`steps_per_epoch`),_;g!=null&&(_=Ny(0,g)),a??=1;let{callbackList:v,history:y}=Ox(o,a,i,d,g,f,r,m,u);v.setModel(this),this.history=y,await v.onTrainBegin(),this.stopTraining_=!1;for(let a=d;a<i;++a){await v.onEpochBegin(a);let i={};if(f!=null)throw new Lv(`stepsPerEpoch mode is not implemented yet.`);{if(l===`batch`)throw new Lv(`batch shuffling is not implemneted yet`);l&&h(_);let a=Af(_),o=GS(g,r);for(let l=0;l<o.length;++l){let u={};if(await v.onBatchBegin(l,u),F(()=>{let d=o[l][0],f=o[l][1],p=Hy(a,d,f-d);u.batch=l,u.size=f-d;let h=e(WS(t,p));for(let e=0;e<n.length;++e){let t=n[e],r=h[e];u[t]=r,ba(r)}if(l===o.length-1&&m){let e=this.testLoop(s,c,r);for(let t=0;t<n.length;++t){let r=n[t],a=e[t];ba(a),i[`val_`+r]=a}}}),await v.onBatchEnd(l,u),yx(u),this.stopTraining_)break}a.dispose()}if(await v.onEpochEnd(a,i),this.stopTraining_)break}return await v.onTrainEnd(),await this.history.syncData(),this.history}async fitDataset(e,t){return LS(this,e,t)}async trainOnBatch(e,t){let n=await this.standardizeUserData(e,t),r=n[0],i=n[1],a=this.makeTrainFunction()(r.concat(i)),o=[];for(let e of a){let t=await e.data();o.push(t[0])}return ya(a),qS(n[0],e),qS(n[1],t),Uv(o)}getNamedWeights(e){let t=[],n=e!=null&&e.trainableOnly,r=n?this.trainableWeights:this.weights,i=this.getWeights(n);for(let e=0;e<r.length;++e)(!n||r[e].trainable)&&t.push({name:r[e].originalName,tensor:i[e]});return t}set stopTraining(e){this.stopTraining_=e}get stopTraining(){return this.stopTraining_}get optimizer(){return this.optimizer_}set optimizer(e){this.optimizer_!==e&&(this.optimizer_=e,this.isOptimizerOwned=!1)}dispose(){let e=super.dispose();if(e.refCountAfterDispose===0&&this.optimizer!=null&&this.isOptimizerOwned){let t=va().numTensors;this.optimizer_.dispose(),e.numDisposedVariables+=t-va().numTensors}return e}getLossIdentifiers(){let e;if(typeof this.loss==`string`)e=Gv(this.loss);else if(Array.isArray(this.loss)){for(let e of this.loss)if(typeof e!=`string`)throw Error(`Serialization of non-string loss is not supported.`);e=this.loss.map(e=>Gv(e))}else{let t=Object.keys(this.loss);e={};let n=this.loss;for(let r of t)if(typeof n[r]==`string`)e[r]=Gv(n[r]);else throw Error(`Serialization of non-string loss is not supported.`)}return e}getMetricIdentifiers(){if(typeof this.metrics==`string`||typeof this.metrics==`function`)return[Gv(fS(this.metrics))];if(Array.isArray(this.metrics))return this.metrics.map(e=>Gv(fS(e)));{let e={};for(let t in this.metrics)e[t]=Gv(fS(this.metrics[t]));return e}}getTrainingConfig(){return{loss:this.getLossIdentifiers(),metrics:this.getMetricIdentifiers(),optimizer_config:{class_name:this.optimizer.getClassName(),config:this.optimizer.getConfig()}}}loadTrainingConfig(e){if(e.weighted_metrics!=null)throw Error(`Loading weight_metrics is not supported yet.`);if(e.loss_weights!=null)throw Error(`Loading loss_weights is not supported yet.`);if(e.sample_weight_mode!=null)throw Error(`Loading sample_weight_mode is not supported yet.`);let t=kx(wS(e.optimizer_config)),n;if(typeof e.loss==`string`)n=Kv(e.loss);else if(Array.isArray(e.loss))n=e.loss.map(e=>Kv(e));else if(e.loss!=null){n={};for(let t in e.loss)n[t]=Kv(e.loss[t])}let r;if(Array.isArray(e.metrics))r=e.metrics.map(e=>Kv(e));else if(e.metrics!=null){r={};for(let t in e.metrics)r[t]=Kv(e.metrics[t])}this.compile({loss:n,metrics:r,optimizer:t})}async save(e,t){if(typeof e==`string`){let t=Ia(e);if(t.length===0)throw new K(`Cannot find any save handlers for URL '${e}'`);if(t.length>1)throw new K(`Found more than one (${t.length}) save handlers for URL '${e}'`);e=t[0]}if(e.save==null)throw new K("LayersModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");let n=await Da(this.getNamedWeights(t)),r={modelTopology:this.toJSON(null,!1),format:nC,generatedBy:`TensorFlow.js tfjs-layers v${ES}`,convertedBy:null};if(t!=null&&t.includeOptimizer&&this.optimizer!=null){r.trainingConfig=this.getTrainingConfig();let{data:e,specs:t}=await Da(await this.optimizer.getWeights(),`optimizer`);n.specs.push(...t),n.data=Na([n.data,e])}return this.userDefinedMetadata!=null&&(hS(this.userDefinedMetadata,this.name,!0),r.userDefinedMetadata=this.userDefinedMetadata),r.weightData=n.data,r.weightSpecs=n.specs,e.save(r)}setUserDefinedMetadata(e){hS(e,this.name),this.userDefinedMetadata=e}getUserDefinedMetadata(){return this.userDefinedMetadata}};rC.className=`Model`,G(rC);var iC=class extends rC{};iC.className=`Functional`,G(iC);var aC=class e extends rC{constructor(e){if(super({inputs:[],outputs:[]}),e||={},this.trainable=!0,this.built=!1,this.name=e.name==null?uy(`sequential_`):e.name,e.layers!=null)for(let t of e.layers)this.add(t)}checkShape(e){if(e.inboundNodes[0].outputTensors[0].shape.some(e=>e<0))throw new K(`Negative dimension size caused by adding layer ${e.name} with input shape [${e.inboundNodes[0].inputTensors[0].shape}]`)}add(t){let n=t instanceof e||t instanceof rC,r;if(n){if(r=t,r.outputs.length!==1)throw new K(`All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.`);if(r.inputs.length!==1)throw new K(`All layers in a Sequential model should have a single input tensor. For multi-input layers, use the functional API.`)}if(this.outputs.length===0){if(t.inboundNodes.length===0){if(t.batchInputShape==null)throw new K("The first layer in a Sequential model must get an `inputShape` or `batchInputShape` argument.");let e=Zb({batchShape:t.batchInputShape,dtype:t.dtype,name:t.name+`_input`});t.apply(e)}if(n)this.outputs=r.outputs,this.inputs=r.inputs;else{if(t.inboundNodes.length!==1)throw new K(`A layer added to a Sequential model must not already be connected somewhere else. LayersModel received layer ${t.name} which has ${t.inboundNodes.length} pre-existing inbound connections.`);if(t.inboundNodes[0].outputTensors.length!==1)throw new K(`All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.`);this.checkShape(t),this.outputs=[t.inboundNodes[0].outputTensors[0]],this.inputs=qb(this.outputs[0])}this.inboundNodes=[],new Hb({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:Bv(null,this.inputs.length),outputMasks:[null],inputShapes:this.inputs.map(e=>e.shape),outputShapes:this.outputs[0].shape})}else{let e=t.apply(this.outputs[0]);if(Array.isArray(e))throw TypeError(`All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.`);this.checkShape(t),this.outputs=[e],this.inboundNodes[0].outputTensors=this.outputs,this.inboundNodes[0].outputShapes=[this.outputs[0].shape]}this.layers.push(t),this.built=!1}pop(){if(this.layers.length===0)throw TypeError(`There are no layers in the model.`);if(this.layers.pop(),this.layers.length===0)this.outputs=[],this.inboundNodes=[],this.outboundNodes=[];else{let e=this.layers.length-1;this.layers[e].outboundNodes=[],this.outputs=[this.layers[e].output],this.inboundNodes[0].outputTensors=this.outputs,this.inboundNodes[0].outputShapes=[this.outputs[0].shape]}}call(e,t){return this.model??this.build(),this.model.call(e,t)}build(e){if(Mb(e),this.inputs.length===0||this.outputs.length===0)throw TypeError(`Sequential model cannot be built: model is empty. Add some layers first.`);this.model=new rC({inputs:this.inputs,outputs:this.outputs[0],name:this.name+`_model`}),this.model.trainable=this.trainable,this.supportsMasking=this.model.supportsMasking,this.inputLayers=this.model.inputLayers,this.inputLayersNodeIndices=this.model.inputLayersNodeIndices,this.inputLayersTensorIndices=this.model.inputLayersTensorIndices,this.outputLayers=this.model.outputLayers,this.outputLayersNodeIndices=this.model.outputLayersNodeIndices,this.outputLayersTensorIndices=this.model.outputLayersTensorIndices,this.nodesByDepth=this.model.nodesByDepth,this.containerNodes=this.model.containerNodes,this.outputNames=this.model.outputNames,this.inputNames=this.model.inputNames,this.built=!0}countParams(){return this.built||this.build(),super.countParams()}summary(e,t,n=console.log){this.built||this.build(),super.summary(e,t,n)}setWeights(e){this.model??this.build(),this.model.setWeights(e)}evaluate(e,t,n={}){if(!this.built)throw new Iv(`The model needs to be compiled before being used.`);return this.model.evaluate(e,t,n)}async evaluateDataset(e,t){if(!this.built)throw new Iv(`The model needs to be compiled before being used.`);return this.model.evaluateDataset(e,t)}predict(e,t={}){return this.model??this.build(),this.model.predict(e,t)}predictOnBatch(e){return this.model??this.build(),this.model.predictOnBatch(e)}compile(e){this.build(),this.model.compile(e),this.optimizer_=this.model.optimizer,this.isOptimizerOwned=this.model.isOptimizerOwned,this.loss=this.model.loss,this.metrics=this.model.metrics,this.metricsTensors=this.model.metricsTensors,this.metricsNames=this.model.metricsNames}get optimizer(){return this.model==null?void 0:this.model.optimizer}set optimizer(e){this.model.optimizer=e}async fit(e,t,n={}){if(!this.built)throw new Iv(`The model needs to be compiled before being used.`);return this.model.fit(e,t,n)}async fitDataset(e,t){if(!this.built)throw new Iv(`The model needs to be compiled before being used.`);return this.model.fitDataset(e,t)}async trainOnBatch(e,t){return this.model.trainOnBatch(e,t)}static fromConfig(t,n,r={},i=!1){let a,o={};if(n instanceof Array){if(n[0].className==null||n[0].className===`Merge`)throw new K(`Legacy serialization format not supported yet.`);a=n}else b(n.layers!=null,()=>`When the config data for a Sequential model is not an Array, it must be an Object that contains the 'layers' field.`),a=n.layers,delete n.layers,o=n;let s=new t(o);if(!(s instanceof e))throw new Lv(`Sequential.fromConfig called on non-Sequential input: ${s}`);for(let e of a){let t=kx(e,void 0,i);i&&t.setFastWeightInitDuringBuild(!0),s.add(t)}return s}set stopTraining(e){if(this.model==null)throw new K(`Cannot set the stopTraining property of a sequential model before it is compiled.`);this.model.stopTraining=e}get stopTraining(){if(this.model==null)throw new K(`Cannot get the stopTraining property of a sequential model before it is compiled.`);return this.model.stopTraining}getConfig(){let e=[];for(let t of this.layers){let n={};n.className=t.getClassName(),n.config=t.getConfig(),e.push(n)}return{name:this.name,layers:e}}};aC.className=`Sequential`,G(aC);var oC=class extends Fm{getConfig(){return{}}},sC=class extends oC{apply(e,t=1){return eb(e,t)}};sC.className=`elu`,G(sC);var cC=class extends oC{apply(e){return Hd(e)}};cC.className=`selu`,G(cC);var lC=class extends oC{apply(e){return Md(e)}};lC.className=`relu`,G(lC);var uC=class extends oC{apply(e){return F(()=>Vu(6,Md(e)))}};uC.className=`relu6`,G(uC);var dC=class extends oC{apply(e){return e}};dC.className=`linear`,G(dC);var fC=class extends oC{apply(e){return Os(e)}};fC.className=`sigmoid`,G(fC);var pC=class extends oC{apply(e){return rb(e)}};pC.className=`hardSigmoid`,G(pC);var mC=class extends oC{apply(e){return mu(e)}};mC.className=`softplus`,G(mC);var hC=class extends oC{apply(e){return tb(e)}};hC.className=`softsign`,G(hC);var gC=class extends oC{apply(e){return Ms(e)}};gC.className=`tanh`,G(gC);var _C=class extends oC{apply(e,t=-1){return sf(e,t)}};_C.className=`softmax`,G(_C);var vC=class extends oC{apply(e,t=-1){return yu(e,t)}};vC.className=`logSoftmax`,G(vC);var yC=class extends oC{apply(e){return F(()=>F(()=>z(e,z(.5,L(1,el(R(e,Math.sqrt(2))))))))}};yC.className=`gelu`,G(yC);var bC=class extends oC{apply(e){return F(()=>z(.5,z(e,L(1,Ms(z(gl(R(2,Math.PI)),L(e,z(.044715,ml(e,3)))))))))}};bC.className=`gelu_new`,G(bC);var xC=class extends oC{apply(e){return F(()=>z(e,Ms(mu(e))))}};xC.className=`mish`,G(xC);var SC=class extends oC{apply(e,t=1){return F(()=>z(Os(z(e,t)),e))}};SC.className=`swish`,G(SC);function CC(e){return e.getClassName()}function wC(e,t={}){return Xv(e,Im.getMap().classNameMap,t,`activation`)}function TC(e){if(e==null){let e={};return e.className=`linear`,e.config={},wC(e)}if(typeof e==`string`){let t={};return t.className=e,t.config={},wC(t)}return e instanceof oC?e:wC(e)}function EC(e){if(e!=null&&typeof e!=`object`)throw Error(`Argument to L1L2 regularizer's constructor is expected to be an object, but received: ${e}`)}var DC=class extends Fm{},OC=class extends DC{constructor(e){super(),EC(e),this.l1=e==null||e.l1==null?.01:e.l1,this.l2=e==null||e.l2==null?.01:e.l2,this.hasL1=this.l1!==0,this.hasL2=this.l2!==0}apply(e){return F(()=>{let t=Ru([1]);return this.hasL1&&(t=L(t,U(z(this.l1,Oo(e))))),this.hasL2&&(t=L(t,U(z(this.l2,Zy(e))))),B(t,[])})}getConfig(){return{l1:this.l1,l2:this.l2}}static fromConfig(e,t){return new e({l1:t.l1,l2:t.l2})}};OC.className=`L1L2`,G(OC);var kC={l1l2:`L1L2`};function AC(e){return Jv(e)}function jC(e,t={}){return Xv(e,Im.getMap().classNameMap,t,`regularizer`)}function MC(e){return e==null?null:typeof e==`string`?jC({className:e in kC?kC[e]:e,config:{}}):e instanceof DC?e:jC(e)}var NC=class extends Wb{constructor(e){super(e??{}),this.supportsMasking=!0,e!=null&&(this.maxValue=e.maxValue)}call(e,t){e=q(e);let n=Md(e);return this.maxValue!=null&&(n=Qs(n,0,this.maxValue)),n}computeOutputShape(e){return e}getConfig(){let e={maxValue:this.maxValue},t=super.getConfig();return Object.assign(e,t),e}};NC.className=`ReLU`,G(NC);var PC=class extends Wb{constructor(e){super(e??{}),this.DEFAULT_ALPHA=.3,e??={},this.alpha=e.alpha==null?this.DEFAULT_ALPHA:e.alpha}call(e,t){return Ql(q(e),this.alpha)}computeOutputShape(e){return e}getConfig(){let e={alpha:this.alpha},t=super.getConfig();return Object.assign(e,t),e}};PC.className=`LeakyReLU`,G(PC);var FC=class extends Wb{constructor(e){if(super(e??{}),this.DEFAULT_ALPHA_INITIALIZER=`zeros`,e??={},this.supportsMasking=!0,this.alphaInitializer=kb(e.alphaInitializer||this.DEFAULT_ALPHA_INITIALIZER),this.alphaRegularizer=MC(e.alphaRegularizer),this.alphaConstraint=_x(e.alphaConstraint),e.sharedAxes==null)this.sharedAxes=null;else if(Array.isArray(e.sharedAxes))this.sharedAxes=e.sharedAxes;else if(typeof e.sharedAxes==`number`)this.sharedAxes=[e.sharedAxes];else throw new K(`Expected sharedAxes to be a number or an array of numbers, but got ${e.sharedAxes}`)}build(e){e=Mb(e);let t=e.slice(1);if(this.sharedAxes!=null)for(let e of this.sharedAxes)t[e-1]=1;this.alpha=this.addWeight(`alpha`,t,`float32`,this.alphaInitializer,this.alphaRegularizer,!0,this.alphaConstraint);let n={};if(this.sharedAxes!=null)for(let t=1;t<e.length;++t)n[t]=e[t];this.inputSpec=[new zb({ndim:e.length,axes:n})],this.built=!0}call(e,t){return e=q(e),ld(e,this.alpha.read())}getConfig(){let e={alphaInitializer:Ob(this.alphaInitializer),alphaRegularizer:AC(this.alphaRegularizer),alphaConstraint:hx(this.alphaConstraint),sharedAxes:this.sharedAxes},t=super.getConfig();return Object.assign(e,t),e}};FC.className=`PReLU`,G(FC);var IC=class extends Wb{constructor(e){if(super(e??{}),this.DEFAULT_ALPHA=1,e??={},e.alpha!=null&&e.alpha!==this.DEFAULT_ALPHA)throw new Lv(`Non-default alpha value (${e.alpha}) is not supported by the ELU layer yet.`);this.alpha=e.alpha==null?this.DEFAULT_ALPHA:e.alpha}call(e,t){return Qc(q(e))}computeOutputShape(e){return e}getConfig(){let e={alpha:this.alpha},t=super.getConfig();return Object.assign(e,t),e}};IC.className=`ELU`,G(IC);var LC=class extends Wb{constructor(e){super(e??{}),this.DEFAULT_THETA=1,e??={},this.theta=e.theta==null?this.DEFAULT_THETA:e.theta}call(e,t){let n=q(e);return z(n,I(Bl(n,this.theta),`float32`))}computeOutputShape(e){return e}getConfig(){let e={theta:this.theta},t=super.getConfig();return Object.assign(e,t),e}};LC.className=`ThresholdedReLU`,G(LC);var RC=class extends Wb{constructor(e){super(e??{}),this.DEFAULT_AXIS=1,e??={},this.softmax=new _C().apply,this.axis=e.axis==null?this.DEFAULT_AXIS:e.axis}call(e,t){return F(()=>{let n=q(e),r=t.mask;if(r!=null){let e=z(W(zu(n.shape),I(r,n.dtype)),H(-1e9));n=L(n,e)}return this.axis instanceof Array?this.axis.length>1?El(W(n,xu(n,this.axis,!0))):this.softmax(n,this.axis[0]):this.softmax(n,this.axis)})}computeOutputShape(e){return e}getConfig(){let e={axis:this.axis},t=super.getConfig();return Object.assign(e,t),e}};RC.className=`Softmax`,G(RC);function zC(e,t,n){if(typeof e==`number`)return Bv(e,t);if(e.length!==t)throw new K(`The ${n} argument must be an integer or tuple of ${t} integers. Received: ${e.length} elements.`);for(let r=0;r<t;++r){let i=e[r];if(!ky(i))throw new K(`The ${n} argument must be an integer or tuple of ${t} integers. Received: ${JSON.stringify(e)} including a non-integer number ${i}`)}return e}function BC(e,t,n,r,i=1){if(e==null)return e;let a=t+(t-1)*(i-1),o;return o=n===`same`?e:e-a+1,Math.floor((o+r-1)/r)}function VC(e,t,n,r){if(e==null)return null;if(r===`valid`)e=e*t+My([n-t,0]);else if(r===`same`)e*=t;else throw new K(`Unsupport padding mode: ${r}.`);return e}function HC(e,t){return F(()=>(_y(t),t===`channelsFirst`?Yf(e,[0,2,3,1]):e))}function UC(e,t){return F(()=>(_y(t),t===`channelsFirst`?Yf(e,[0,2,3,4,1]):e))}function WC(e,t,n,r=1,i=`valid`,a,o=1){return F(()=>{if(a??=Iy(),_y(a),e.shape.length!==3)throw new K(`The input of a conv1dWithBias operation should be 3, but is ${e.shape.length} instead.`);if(t.shape.length!==3)throw new K(`The kernel for a conv1dWithBias operation should be 3, but is ${t.shape.length} instead`);if(n!=null&&n.shape.length!==1)throw new K(`The bias for a conv1dWithBias operation should be 1, but is ${n.shape.length} instead`);if(a===`channelsFirst`&&(e=Yf(e,[0,2,1])),i===`causal`)throw new Lv(`The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.`);let s=uc(e,t,r,i===`same`?`same`:`valid`,`NWC`,o);return n!=null&&(s=$y(s,n)),s})}function GC(e,t,n,r=[1,1],i=`valid`,a,o,s=null){return F(()=>{if(a??=Iy(),_y(a),e.rank!==3&&e.rank!==4)throw new K(`conv2dWithBiasActivation expects input to be of rank 3 or 4, but received ${e.rank}.`);if(t.rank!==3&&t.rank!==4)throw new K(`conv2dWithBiasActivation expects kernel to be of rank 3 or 4, but received ${e.rank}.`);let c=HC(e,a);if(i===`causal`)throw new Lv(`The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.`);return c=op({x:c,filter:t,strides:r,pad:i===`same`?`same`:`valid`,dilations:o,dataFormat:`NHWC`,bias:n,activation:s}),a===`channelsFirst`&&(c=Yf(c,[0,3,1,2])),c})}function KC(e,t,n,r=[1,1,1],i=`valid`,a,o){return F(()=>{if(a??=Iy(),_y(a),e.rank!==4&&e.rank!==5)throw new K(`conv3dWithBias expects input to be of rank 4 or 5, but received ${e.rank}.`);if(t.rank!==4&&t.rank!==5)throw new K(`conv3dWithBias expects kernel to be of rank 4 or 5, but received ${e.rank}.`);let s=UC(e,a);if(i===`causal`)throw new Lv(`The support for CAUSAL padding mode in conv3dWithBias is not implemented yet.`);return s=gc(s,t,r,i===`same`?`same`:`valid`,`NDHWC`,o),n!=null&&(s=$y(s,n)),a===`channelsFirst`&&(s=Yf(s,[0,4,1,2,3])),s})}var qC=class e extends Wb{constructor(t,n){if(super(n),this.bias=null,this.DEFAULT_KERNEL_INITIALIZER=`glorotNormal`,this.DEFAULT_BIAS_INITIALIZER=`zeros`,e.verifyArgs(n),this.rank=t,ry(this.rank,`rank`),this.rank!==1&&this.rank!==2&&this.rank!==3)throw new Lv(`Convolution layer for rank other than 1, 2, or 3 (${this.rank}) is not implemented yet.`);if(this.kernelSize=zC(n.kernelSize,t,`kernelSize`),this.strides=zC(n.strides==null?1:n.strides,t,`strides`),this.padding=n.padding==null?`valid`:n.padding,yy(this.padding),this.dataFormat=n.dataFormat==null?`channelsLast`:n.dataFormat,_y(this.dataFormat),this.activation=TC(n.activation),this.useBias=n.useBias==null||n.useBias,this.biasInitializer=kb(n.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.biasConstraint=_x(n.biasConstraint),this.biasRegularizer=MC(n.biasRegularizer),this.activityRegularizer=MC(n.activityRegularizer),this.dilationRate=zC(n.dilationRate==null?1:n.dilationRate,t,`dilationRate`),this.rank===1&&Array.isArray(this.dilationRate)&&this.dilationRate.length!==1)throw new K(`dilationRate must be a number or an array of a single number for 1D convolution, but received ${JSON.stringify(this.dilationRate)}`);if(this.rank===2){if(typeof this.dilationRate==`number`)this.dilationRate=[this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==2)throw new K(`dilationRate must be a number or array of two numbers for 2D convolution, but received ${JSON.stringify(this.dilationRate)}`)}else if(this.rank===3){if(typeof this.dilationRate==`number`)this.dilationRate=[this.dilationRate,this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==3)throw new K(`dilationRate must be a number or array of three numbers for 3D convolution, but received ${JSON.stringify(this.dilationRate)}`)}}static verifyArgs(e){if(Vv(`kernelSize`in e,`required key 'kernelSize' not in config`),typeof e.kernelSize!=`number`&&!ny(e.kernelSize,`number`,1,3))throw new K(`BaseConv expects config.kernelSize to be number or number[] with length 1, 2, or 3, but received ${JSON.stringify(e.kernelSize)}.`)}getConfig(){let e={kernelSize:this.kernelSize,strides:this.strides,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,activation:CC(this.activation),useBias:this.useBias,biasInitializer:Ob(this.biasInitializer),biasRegularizer:AC(this.biasRegularizer),activityRegularizer:AC(this.activityRegularizer),biasConstraint:hx(this.biasConstraint)},t=super.getConfig();return Object.assign(e,t),e}},JC=class e extends qC{constructor(t,n){super(t,n),this.kernel=null,e.verifyArgs(n),this.filters=n.filters,ry(this.filters,`filters`),this.kernelInitializer=kb(n.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.kernelConstraint=_x(n.kernelConstraint),this.kernelRegularizer=MC(n.kernelRegularizer)}build(e){e=Mb(e);let t=this.dataFormat===`channelsFirst`?1:e.length-1;if(e[t]==null)throw new K(`The channel dimension of the input should be defined. Found ${e[t]}`);let n=e[t],r=this.kernelSize.concat([n,this.filters]);this.kernel=this.addWeight(`kernel`,r,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight(`bias`,[this.filters],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[{ndim:this.rank+2,axes:{[t]:n}}],this.built=!0}call(e,t){return F(()=>{e=q(e);let t,n=this.bias==null?null:this.bias.read(),r=oy(this.activation.getClassName());if(r!=null&&this.rank===2)t=GC(e,this.kernel.read(),n,this.strides,this.padding,this.dataFormat,this.dilationRate,r);else{if(this.rank===1)t=WC(e,this.kernel.read(),n,this.strides[0],this.padding,this.dataFormat,this.dilationRate[0]);else if(this.rank===2)t=GC(e,this.kernel.read(),n,this.strides,this.padding,this.dataFormat,this.dilationRate);else if(this.rank===3)t=KC(e,this.kernel.read(),n,this.strides,this.padding,this.dataFormat,this.dilationRate);else throw new Lv(`convolutions greater than 3D are not implemented yet.`);this.activation!=null&&(t=this.activation.apply(t))}return t})}computeOutputShape(e){e=Mb(e);let t=[],n=this.dataFormat===`channelsLast`?e.slice(1,e.length-1):e.slice(2);for(let e=0;e<n.length;++e){let r=BC(n[e],this.kernelSize[e],this.padding,this.strides[e],typeof this.dilationRate==`number`?this.dilationRate:this.dilationRate[e]);t.push(r)}let r=[e[0]];return this.dataFormat===`channelsLast`?(r=r.concat(t),r.push(this.filters)):(r.push(this.filters),r=r.concat(t)),r}getConfig(){let e={filters:this.filters,kernelInitializer:Ob(this.kernelInitializer),kernelRegularizer:AC(this.kernelRegularizer),kernelConstraint:hx(this.kernelConstraint)},t=super.getConfig();return Object.assign(e,t),e}static verifyArgs(e){if(!(`filters`in e)||typeof e.filters!=`number`||e.filters<1)throw new K(`Convolution layer expected config.filters to be a 'number' > 0 but got ${JSON.stringify(e.filters)}`)}},YC=class e extends JC{constructor(t){super(2,t),e.verifyArgs(t)}getConfig(){let e=super.getConfig();return delete e.rank,e}static verifyArgs(e){if(typeof e.kernelSize!=`number`&&!ny(e.kernelSize,`number`,1,2))throw new K(`Conv2D expects config.kernelSize to be number or number[] with length 1 or 2, but received ${JSON.stringify(e.kernelSize)}.`)}};YC.className=`Conv2D`,G(YC);var XC=class e extends JC{constructor(t){super(3,t),e.verifyArgs(t)}getConfig(){let e=super.getConfig();return delete e.rank,e}static verifyArgs(e){if(typeof e.kernelSize!=`number`&&(!Array.isArray(e.kernelSize)||e.kernelSize.length!==1&&e.kernelSize.length!==3))throw new K(`Conv3D expects config.kernelSize to be number or [number, number, number], but received ${JSON.stringify(e.kernelSize)}.`)}};XC.className=`Conv3D`,G(XC);var ZC=class extends YC{constructor(e){if(super(e),this.inputSpec=[new zb({ndim:4})],this.padding!==`same`&&this.padding!==`valid`)throw new K(`Conv2DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(e){if(e=Mb(e),e.length!==4)throw new K(`Input should have rank 4; Received input shape: `+JSON.stringify(e));let t=this.dataFormat===`channelsFirst`?1:e.length-1;if(e[t]==null)throw new K("The channel dimension of the inputs should be defined. Found `None`.");let n=e[t],r=this.kernelSize.concat([this.filters,n]);this.kernel=this.addWeight(`kernel`,r,`float32`,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight(`bias`,[this.filters],`float32`,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new zb({ndim:4,axes:{[t]:n}})],this.built=!0}call(e,t){return F(()=>{let t=q(e);if(t.shape.length!==4)throw new K(`Conv2DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${t.shape.length}`);let n=t.shape,r=n[0],i,a;this.dataFormat===`channelsFirst`?(i=2,a=3):(i=1,a=2);let o=n[i],s=n[a],c=this.kernelSize[0],l=this.kernelSize[1],u=this.strides[0],d=this.strides[1],f=[r,VC(o,u,c,this.padding),VC(s,d,l,this.padding),this.filters];this.dataFormat!==`channelsLast`&&(t=Yf(t,[0,2,3,1]));let p=mc(t,this.kernel.read(),f,this.strides,this.padding);return this.dataFormat!==`channelsLast`&&(p=Yf(p,[0,3,1,2])),this.bias!=null&&(p=$y(p,this.bias.read(),this.dataFormat)),this.activation!=null&&(p=this.activation.apply(p)),p})}computeOutputShape(e){e=Mb(e);let t=e.slice(),n,r,i;this.dataFormat===`channelsFirst`?(n=1,r=2,i=3):(n=3,r=1,i=2);let a=this.kernelSize[0],o=this.kernelSize[1],s=this.strides[0],c=this.strides[1];return t[n]=this.filters,t[r]=VC(t[r],s,a,this.padding),t[i]=VC(t[i],c,o,this.padding),t}getConfig(){let e=super.getConfig();return delete e.dilationRate,e}};ZC.className=`Conv2DTranspose`,G(ZC);var QC=class extends XC{constructor(e){if(super(e),this.inputSpec=[new zb({ndim:5})],this.padding!==`same`&&this.padding!==`valid`)throw new K(`Conv3DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(e){if(e=Mb(e),e.length!==5)throw new K(`Input should have rank 5; Received input shape: `+JSON.stringify(e));let t=this.dataFormat===`channelsFirst`?1:e.length-1;if(e[t]==null)throw new K("The channel dimension of the inputs should be defined. Found `None`.");let n=e[t],r=this.kernelSize.concat([this.filters,n]);this.kernel=this.addWeight(`kernel`,r,`float32`,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight(`bias`,[this.filters],`float32`,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new zb({ndim:5,axes:{[t]:n}})],this.built=!0}call(e,t){return F(()=>{let t=q(e);if(t.shape.length!==5)throw new K(`Conv3DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${t.shape.length}`);let n=t.shape,r=n[0],i,a,o;this.dataFormat===`channelsFirst`?(o=2,i=3,a=4):(o=1,i=2,a=3);let s=n[o],c=n[i],l=n[a],u=this.kernelSize[0],d=this.kernelSize[1],f=this.kernelSize[2],p=this.strides[0],m=this.strides[1],h=this.strides[2],g=[r,VC(s,p,u,this.padding),VC(c,m,d,this.padding),VC(l,h,f,this.padding),this.filters];this.dataFormat!==`channelsLast`&&(t=Yf(t,[0,2,3,4,1]));let _=bc(t,this.kernel.read(),g,this.strides,this.padding);return this.dataFormat!==`channelsLast`&&(_=Yf(_,[0,4,1,2,3])),this.bias!==null&&(_=$y(_,this.bias.read(),this.dataFormat)),this.activation!==null&&(_=this.activation.apply(_)),_})}computeOutputShape(e){e=Mb(e);let t=e.slice(),n,r,i,a;this.dataFormat===`channelsFirst`?(n=1,r=2,i=3,a=4):(n=4,r=1,i=2,a=3);let o=this.kernelSize[0],s=this.kernelSize[1],c=this.kernelSize[2],l=this.strides[0],u=this.strides[1],d=this.strides[2];return t[n]=this.filters,t[r]=VC(t[r],l,o,this.padding),t[i]=VC(t[i],u,s,this.padding),t[a]=VC(t[a],d,c,this.padding),t}getConfig(){let e=super.getConfig();return delete e.dilationRate,e}};QC.className=`Conv3DTranspose`,G(QC);var $C=class extends JC{constructor(e,t){if(super(e,t),this.DEFAULT_DEPTHWISE_INITIALIZER=`glorotUniform`,this.DEFAULT_POINTWISE_INITIALIZER=`glorotUniform`,this.depthwiseKernel=null,this.pointwiseKernel=null,t.filters==null)throw new K("The `filters` configuration field is required by SeparableConv, but is unspecified.");if(t.kernelInitializer!=null||t.kernelRegularizer!=null||t.kernelConstraint!=null)throw new K(`Fields kernelInitializer, kernelRegularizer and kernelConstraint are invalid for SeparableConv2D. Use depthwiseInitializer, depthwiseRegularizer, depthwiseConstraint, pointwiseInitializer, pointwiseRegularizer and pointwiseConstraint instead.`);if(t.padding!=null&&t.padding!==`same`&&t.padding!==`valid`)throw new K(`SeparableConv${this.rank}D supports only padding modes: 'same' and 'valid', but received ${JSON.stringify(t.padding)}`);this.depthMultiplier=t.depthMultiplier==null?1:t.depthMultiplier,this.depthwiseInitializer=kb(t.depthwiseInitializer||this.DEFAULT_DEPTHWISE_INITIALIZER),this.depthwiseRegularizer=MC(t.depthwiseRegularizer),this.depthwiseConstraint=_x(t.depthwiseConstraint),this.pointwiseInitializer=kb(t.depthwiseInitializer||this.DEFAULT_POINTWISE_INITIALIZER),this.pointwiseRegularizer=MC(t.pointwiseRegularizer),this.pointwiseConstraint=_x(t.pointwiseConstraint)}build(e){if(e=Mb(e),e.length<this.rank+2)throw new K(`Inputs to SeparableConv${this.rank}D should have rank ${this.rank+2}, but received input shape: ${JSON.stringify(e)}`);let t=this.dataFormat===`channelsFirst`?1:e.length-1;if(e[t]==null||e[t]<0)throw new K(`The channel dimension of the inputs should be defined, but found ${JSON.stringify(e[t])}`);let n=e[t],r=this.kernelSize.concat([n,this.depthMultiplier]),i=[];for(let e=0;e<this.rank;++e)i.push(1);i.push(n*this.depthMultiplier,this.filters),this.depthwiseKernel=this.addWeight(`depthwise_kernel`,r,`float32`,this.depthwiseInitializer,this.depthwiseRegularizer,!0,this.depthwiseConstraint),this.pointwiseKernel=this.addWeight(`pointwise_kernel`,i,`float32`,this.pointwiseInitializer,this.pointwiseRegularizer,!0,this.pointwiseConstraint),this.bias=this.useBias?this.addWeight(`bias`,[this.filters],`float32`,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):null,this.inputSpec=[new zb({ndim:this.rank+2,axes:{[t]:n}})],this.built=!0}call(e,t){return F(()=>{e=q(e);let t;if(this.rank===1)throw new Lv(`1D separable convolution is not implemented yet.`);return this.rank===2&&(this.dataFormat===`channelsFirst`&&(e=Yf(e,[0,2,3,1])),t=Wd(e,this.depthwiseKernel.read(),this.pointwiseKernel.read(),this.strides,this.padding,this.dilationRate,`NHWC`)),this.useBias&&(t=$y(t,this.bias.read(),this.dataFormat)),this.activation!=null&&(t=this.activation.apply(t)),this.dataFormat===`channelsFirst`&&(t=Yf(t,[0,3,1,2])),t})}getConfig(){let e=super.getConfig();return delete e.rank,delete e.kernelInitializer,delete e.kernelRegularizer,delete e.kernelConstraint,e.depthwiseInitializer=Ob(this.depthwiseInitializer),e.pointwiseInitializer=Ob(this.pointwiseInitializer),e.depthwiseRegularizer=AC(this.depthwiseRegularizer),e.pointwiseRegularizer=AC(this.pointwiseRegularizer),e.depthwiseConstraint=hx(this.depthwiseConstraint),e.pointwiseConstraint=hx(this.pointwiseConstraint),e}};$C.className=`SeparableConv`;var ew=class extends $C{constructor(e){super(2,e)}};ew.className=`SeparableConv2D`,G(ew);var tw=class e extends JC{constructor(t){super(1,t),e.verifyArgs(t),this.inputSpec=[{ndim:3}]}getConfig(){let e=super.getConfig();return delete e.rank,delete e.dataFormat,e}static verifyArgs(e){if(typeof e.kernelSize!=`number`&&!ny(e.kernelSize,`number`,1,1))throw new K(`Conv1D expects config.kernelSize to be number or number[] with length 1, but received ${JSON.stringify(e.kernelSize)}.`)}};tw.className=`Conv1D`,G(tw);var nw=class extends Wb{constructor(e){super(e),this.cropping=typeof e.cropping==`number`?[[e.cropping,e.cropping],[e.cropping,e.cropping]]:typeof e.cropping[0]==`number`?[[e.cropping[0],e.cropping[0]],[e.cropping[1],e.cropping[1]]]:e.cropping,this.dataFormat=e.dataFormat===void 0?`channelsLast`:e.dataFormat,this.inputSpec=[{ndim:4}]}computeOutputShape(e){return this.dataFormat===`channelsFirst`?[e[0],e[1],e[2]-this.cropping[0][0]-this.cropping[0][1],e[3]-this.cropping[1][0]-this.cropping[1][1]]:[e[0],e[1]-this.cropping[0][0]-this.cropping[0][1],e[2]-this.cropping[1][0]-this.cropping[1][1],e[3]]}call(e,t){return F(()=>(e=q(e),this.dataFormat===`channelsLast`?Wy(Wy(e,this.cropping[0][0],e.shape[1]-this.cropping[0][0]-this.cropping[0][1],2),this.cropping[1][0],e.shape[2]-this.cropping[1][1]-this.cropping[1][0],3):Wy(Wy(e,this.cropping[0][0],e.shape[2]-this.cropping[0][0]-this.cropping[0][1],3),this.cropping[1][0],e.shape[3]-this.cropping[1][1]-this.cropping[1][0],4)))}getConfig(){let e={cropping:this.cropping,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}};nw.className=`Cropping2D`,G(nw);var rw=class extends Wb{constructor(e){super(e),this.DEFAULT_SIZE=[2,2],this.inputSpec=[{ndim:4}],this.size=e.size==null?this.DEFAULT_SIZE:e.size,this.dataFormat=e.dataFormat==null?`channelsLast`:e.dataFormat,_y(this.dataFormat),this.interpolation=e.interpolation==null?`nearest`:e.interpolation,vy(this.interpolation)}computeOutputShape(e){if(this.dataFormat===`channelsFirst`){let t=e[2]==null?null:this.size[0]*e[2],n=e[3]==null?null:this.size[1]*e[3];return[e[0],e[1],t,n]}{let t=e[1]==null?null:this.size[0]*e[1],n=e[2]==null?null:this.size[1]*e[2];return[e[0],t,n,e[3]]}}call(e,t){return F(()=>{let t=q(e),n=t.shape;if(this.dataFormat===`channelsFirst`){t=Yf(t,[0,2,3,1]);let e=this.size[0]*n[2],r=this.size[1]*n[3];return Yf(this.interpolation===`nearest`?Am.resizeNearestNeighbor(t,[e,r]):Am.resizeBilinear(t,[e,r]),[0,3,1,2])}{let e=this.size[0]*n[1],r=this.size[1]*n[2];return this.interpolation===`nearest`?Am.resizeNearestNeighbor(t,[e,r]):Am.resizeBilinear(t,[e,r])}})}getConfig(){let e={size:this.size,dataFormat:this.dataFormat,interpolation:this.interpolation},t=super.getConfig();return Object.assign(e,t),e}};rw.className=`UpSampling2D`,G(rw);function iw(e,t,n=[1,1],r=`valid`,i,a){return F(()=>{i??=Iy(),_y(i);let o=HC(e,i);if(e.rank!==4)throw new K(`Input for depthwiseConv2d is required to be 4-D, but is instead ${e.rank}-D`);if(t.rank!==4)throw new K(`depthwiseKernel is required to be 4-D, but is instead ${t.rank}-D`);return o=Pc(o,t,n,r===`same`?`same`:`valid`,`NHWC`,a),i===`channelsFirst`&&(o=Yf(o,[0,3,1,2])),o})}var aw=class extends qC{constructor(e){super(2,e),this.depthwiseKernel=null,this.depthMultiplier=e.depthMultiplier==null?1:e.depthMultiplier,this.depthwiseInitializer=kb(e.depthwiseInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.depthwiseConstraint=_x(e.depthwiseConstraint),this.depthwiseRegularizer=MC(e.depthwiseRegularizer)}build(e){if(e=Mb(e),e.length<4)throw new K(`Inputs to DepthwiseConv2D should have rank 4. Received input shape: ${JSON.stringify(e)}.`);let t=this.dataFormat===`channelsFirst`?1:3;if(e[t]==null||e[t]<0)throw new K(`The channel dimension of the inputs to DepthwiseConv2D should be defined, but is not (${e[t]}).`);let n=e[t],r=[this.kernelSize[0],this.kernelSize[1],n,this.depthMultiplier];this.depthwiseKernel=this.addWeight(`depthwise_kernel`,r,null,this.depthwiseInitializer,this.depthwiseRegularizer,!0,this.depthwiseConstraint),this.bias=this.useBias?this.addWeight(`bias`,[n*this.depthMultiplier],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):null,this.built=!0}call(e,t){return F(()=>{e=q(e);let t=iw(e,this.depthwiseKernel.read(),this.strides,this.padding,this.dataFormat,null);return this.useBias&&(t=$y(t,this.bias.read(),this.dataFormat)),this.activation!=null&&(t=this.activation.apply(t)),t})}computeOutputShape(e){e=Mb(e);let t=this.dataFormat===`channelsFirst`?e[2]:e[1],n=this.dataFormat===`channelsFirst`?e[3]:e[2],r=this.dataFormat===`channelsFirst`?e[1]*this.depthMultiplier:e[3]*this.depthMultiplier,i=BC(t,this.kernelSize[0],this.padding,this.strides[0]),a=BC(n,this.kernelSize[1],this.padding,this.strides[1]);return this.dataFormat===`channelsFirst`?[e[0],r,i,a]:[e[0],i,a,r]}getConfig(){let e=super.getConfig();return e.depthMultiplier=this.depthMultiplier,e.depthwiseInitializer=Ob(this.depthwiseInitializer),e.depthwiseRegularizer=AC(this.depthwiseRegularizer),e.depthwiseConstraint=hx(this.depthwiseRegularizer),e}};aw.className=`DepthwiseConv2D`,G(aw);function ow(e,t,n,r){if(Array.isArray(e)){if(t!=null||n!=null)throw new K(`When inputs is an array, neither initialState or constants should be provided`);r!=null&&(n=e.slice(e.length-r,e.length),e=e.slice(0,e.length-r)),e.length>1&&(t=e.slice(1,e.length)),e=e[0]}function i(e){return e==null||Array.isArray(e)?e:[e]}return t=i(t),n=i(n),{inputs:e,initialState:t,constants:n}}function sw(e,t,n,r=!1,i,a,o=!1,s=!1){return F(()=>{let c=t.shape.length;if(c<3)throw new K(`Input should be at least 3D, but is ${c}D.`);let l=[1,0].concat(Ny(2,c));if(t=Yf(t,l),a!=null)throw new Lv(`The rnn() functoin of the deeplearn.js backend does not support constants yet.`);o&&console.warn(`Backend rnn(): the unroll = true option is not applicable to the imperative deeplearn.js backend.`),i!=null&&(i=I(I(i,`bool`),`float32`),i.rank===c-1&&(i=Ol(i,-1)),i=Yf(i,l)),r&&(t=Id(t,0),i!=null&&(i=Id(i,0)));let u=[],d,f=n,p=t.shape[0],m=Gf(t),h;i!=null&&(h=Gf(i));for(let t=0;t<p;++t){let n=m[t],r=F(()=>e(n,f));if(i==null)d=r[0],f=r[1];else{let e=F(()=>{let e=h[t],n=W($u(e),e);return{output:L(z(r[0],e),z(f[0],n)),newStates:f.map((t,i)=>L(z(r[1][i],e),z(t,n)))}});d=e.output,f=e.newStates}s&&u.push(d)}let g;return s&&(g=Cf(u,1)),[d,g,f]})}var cw=class e extends Wb{constructor(e){super(e);let t;if(e.cell==null)throw new K(`cell property is missing for the constructor of RNN.`);if(t=Array.isArray(e.cell)?new gw({cells:e.cell}):e.cell,t.stateSize==null)throw new K("The RNN cell should have an attribute `stateSize` (tuple of integers, one integer per RNN state).");this.cell=t,this.returnSequences=e.returnSequences!=null&&e.returnSequences,this.returnState=e.returnState!=null&&e.returnState,this.goBackwards=e.goBackwards!=null&&e.goBackwards,this._stateful=e.stateful!=null&&e.stateful,this.unroll=e.unroll!=null&&e.unroll,this.supportsMasking=!0,this.inputSpec=[new zb({ndim:3})],this.stateSpec=null,this.states_=null,this.numConstants=null,this.keptStates=[]}getStates(){return this.states_==null?Ny(0,Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1).map(e=>null):this.states_}setStates(e){this.states_=e}computeOutputShape(e){Ab(e)&&(e=e[0]),e=e;let t=this.cell.stateSize;Array.isArray(t)||(t=[t]);let n=t[0],r;if(r=this.returnSequences?[e[0],e[1],n]:[e[0],n],this.returnState){let n=[];for(let r of t)n.push([e[0],r]);return[r].concat(n)}return r}computeMask(e,t){return F(()=>{Array.isArray(t)&&(t=t[0]);let e=this.returnSequences?t:null;if(this.returnState){let t=this.states.map(e=>null);return[e].concat(t)}return e})}get states(){if(this.states_==null){let e=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1,t=[];for(let n=0;n<e;++n)t.push(null);return t}return this.states_}set states(e){this.states_=e}build(e){if(this.numConstants!=null)throw new Lv(`Constants support is not implemented in RNN yet.`);Ab(e)&&(e=e[0]),e=e;let t=this.stateful?e[0]:null,n=e.slice(2);this.inputSpec[0]=new zb({shape:[t,null,...n]});let r=[e[0]].concat(e.slice(2));this.cell.build(r);let i;if(i=Array.isArray(this.cell.stateSize)?this.cell.stateSize:[this.cell.stateSize],this.stateSpec!=null){if(!w(this.stateSpec.map(e=>e.shape[e.shape.length-1]),i))throw new K(`An initialState was passed that is not compatible with cell.stateSize. Received stateSpec=${this.stateSpec}; However cell.stateSize is ${this.cell.stateSize}`)}else this.stateSpec=i.map(e=>new zb({shape:[null,e]}));this.stateful&&this.resetStates()}resetStates(e,t=!1){F(()=>{if(!this.stateful)throw new Fv(`Cannot call resetStates() on an RNN Layer that is not stateful.`);let n=this.inputSpec[0].shape[0];if(n==null)throw new K("If an RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batchInputShape` option to your first layer.\n- If using the functional API, specify the batch size by passing a `batchShape` option to your Input layer.");if(this.states_==null)this.states_=Array.isArray(this.cell.stateSize)?this.cell.stateSize.map(e=>Ru([n,e])):[Ru([n,this.cell.stateSize])];else if(e==null)ya(this.states_),this.keptStates!=null&&(ya(this.keptStates),this.keptStates=[]),Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(e=>Ru([n,e])):this.states_[0]=Ru([n,this.cell.stateSize]);else{if(Array.isArray(e)||(e=[e]),e.length!==this.states_.length)throw new K(`Layer ${this.name} expects ${this.states_.length} state(s), but it received ${e.length} state value(s). Input received: ${e}`);t===!0?this.keptStates.push(this.states_.slice()):ya(this.states_);for(let t=0;t<this.states_.length;++t){let r=e[t],i=[n,Array.isArray(this.cell.stateSize)?this.cell.stateSize[t]:this.cell.stateSize];if(!w(r.shape,i))throw new K(`State ${t} is incompatible with layer ${this.name}: expected shape=${i}, received shape=${r.shape}`);this.states_[t]=r}}this.states_=this.states_.map(e=>ba(e.clone()))})}apply(e,t){let n=t==null?null:t.initialState,r=t==null?null:t.constants;t??={};let i=ow(e,n,r,this.numConstants);e=i.inputs,n=i.initialState,r=i.constants;let a=[],o=[];if(n!=null){t.initialState=n,a=a.concat(n),this.stateSpec=[];for(let e of n)this.stateSpec.push(new zb({shape:e.shape}));o=o.concat(this.stateSpec)}if(r!=null&&(t.constants=r,a=a.concat(r),this.numConstants=r.length),a[0]instanceof Bb){let n=[e].concat(a),r=this.inputSpec.concat(o),i=this.inputSpec;this.inputSpec=r;let s=super.apply(n,t);return this.inputSpec=i,s}return super.apply(e,t)}call(e,t){return F(()=>{let n=t==null?null:t.mask,r=t==null?null:t.training,i=t==null?null:t.initialState;e=q(e),i??=this.stateful?this.states_:this.getInitialState(e);let a=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1;if(i.length!==a)throw new K(`RNN Layer has ${a} state(s) but was passed ${i.length} initial state(s).`);this.unroll&&console.warn(`Ignoring unroll = true for RNN layer, due to imperative backend.`);let o={training:r},s=sw((e,t)=>{let n=this.cell.call([e].concat(t),o);return[n[0],n.slice(1)]},e,i,this.goBackwards,n,null,this.unroll,this.returnSequences),c=s[0],l=s[1],u=s[2];this.stateful&&this.resetStates(u,r);let d=this.returnSequences?l:c;return this.returnState?[d].concat(u):d})}getInitialState(e){return F(()=>{let t=Ru(e.shape);return t=U(t,[1,2]),t=Ry(t),Array.isArray(this.cell.stateSize)?this.cell.stateSize.map(e=>e>1?qy(t,[1,e]):t):this.cell.stateSize>1?[qy(t,[1,this.cell.stateSize])]:[t]})}get trainableWeights(){return this.trainable?this.cell.trainableWeights:[]}get nonTrainableWeights(){return this.trainable?this.cell.nonTrainableWeights:this.cell.weights}setFastWeightInitDuringBuild(e){super.setFastWeightInitDuringBuild(e),this.cell!=null&&this.cell.setFastWeightInitDuringBuild(e)}getConfig(){let t=super.getConfig(),n={returnSequences:this.returnSequences,returnState:this.returnState,goBackwards:this.goBackwards,stateful:this.stateful,unroll:this.unroll};this.numConstants!=null&&(n.numConstants=this.numConstants);let r=this.cell.getConfig();return this.getClassName()===e.className&&(n.cell={className:this.cell.getClassName(),config:r}),Object.assign(Object.assign(Object.assign({},r),t),n)}static fromConfig(e,t,n={}){let r=t.cell,i=kx(r,n);return new e(Object.assign(t,{cell:i}))}};cw.className=`RNN`,G(cw);var lw=class extends Wb{},uw=class extends lw{constructor(e){super(e),this.DEFAULT_ACTIVATION=`tanh`,this.DEFAULT_KERNEL_INITIALIZER=`glorotNormal`,this.DEFAULT_RECURRENT_INITIALIZER=`orthogonal`,this.DEFAULT_BIAS_INITIALIZER=`zeros`,this.units=e.units,ry(this.units,`units`),this.activation=TC(e.activation==null?this.DEFAULT_ACTIVATION:e.activation),this.useBias=e.useBias==null||e.useBias,this.kernelInitializer=kb(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=kb(e.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=kb(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelRegularizer=MC(e.kernelRegularizer),this.recurrentRegularizer=MC(e.recurrentRegularizer),this.biasRegularizer=MC(e.biasRegularizer),this.kernelConstraint=_x(e.kernelConstraint),this.recurrentConstraint=_x(e.recurrentConstraint),this.biasConstraint=_x(e.biasConstraint),this.dropout=jy([1,My([0,e.dropout==null?0:e.dropout])]),this.recurrentDropout=jy([1,My([0,e.recurrentDropout==null?0:e.recurrentDropout])]),this.dropoutFunc=e.dropoutFunc,this.stateSize=this.units,this.dropoutMask=null,this.recurrentDropoutMask=null}build(e){e=Mb(e),this.kernel=this.addWeight(`kernel`,[e[e.length-1],this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight(`recurrent_kernel`,[this.units,this.units],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.bias=this.useBias?this.addWeight(`bias`,[this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):null,this.built=!0}call(e,t){return F(()=>{if(e=e,e.length!==2)throw new K(`SimpleRNNCell expects 2 input Tensors, got ${e.length}.`);let n=e[1];e=e[0];let r=t.training!=null&&t.training;0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=_w({ones:()=>$u(e),rate:this.dropout,training:r,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=_w({ones:()=>$u(n),rate:this.recurrentDropout,training:r,dropoutFunc:this.dropoutFunc}));let i,a=this.dropoutMask,o=this.recurrentDropoutMask;i=Yy(a==null?e:z(e,a),this.kernel.read()),this.bias!=null&&(i=$y(i,this.bias.read())),o!=null&&(n=z(n,o));let s=L(i,Yy(n,this.recurrentKernel.read()));return this.activation!=null&&(s=this.activation.apply(s)),[s,s]})}getConfig(){let e=super.getConfig(),t={units:this.units,activation:CC(this.activation),useBias:this.useBias,kernelInitializer:Ob(this.kernelInitializer),recurrentInitializer:Ob(this.recurrentInitializer),biasInitializer:Ob(this.biasInitializer),kernelRegularizer:AC(this.kernelRegularizer),recurrentRegularizer:AC(this.recurrentRegularizer),biasRegularizer:AC(this.biasRegularizer),activityRegularizer:AC(this.activityRegularizer),kernelConstraint:hx(this.kernelConstraint),recurrentConstraint:hx(this.recurrentConstraint),biasConstraint:hx(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout};return Object.assign(Object.assign({},e),t)}};uw.className=`SimpleRNNCell`,G(uw);var dw=class extends cw{constructor(e){e.cell=new uw(e),super(e)}call(e,t){return F(()=>{this.cell.dropoutMask!=null&&(ya(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(ya(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);let n=t==null?null:t.mask,r=t==null?null:t.training,i=t==null?null:t.initialState;return super.call(e,{mask:n,training:r,initialState:i})})}static fromConfig(e,t){return new e(t)}};dw.className=`SimpleRNN`,G(dw);var fw=class extends lw{constructor(e){if(super(e),this.DEFAULT_ACTIVATION=`tanh`,this.DEFAULT_RECURRENT_ACTIVATION=`hardSigmoid`,this.DEFAULT_KERNEL_INITIALIZER=`glorotNormal`,this.DEFAULT_RECURRENT_INITIALIZER=`orthogonal`,this.DEFAULT_BIAS_INITIALIZER=`zeros`,e.resetAfter)throw new K(`GRUCell does not support reset_after parameter set to true.`);this.units=e.units,ry(this.units,`units`),this.activation=TC(e.activation===void 0?this.DEFAULT_ACTIVATION:e.activation),this.recurrentActivation=TC(e.recurrentActivation===void 0?this.DEFAULT_RECURRENT_ACTIVATION:e.recurrentActivation),this.useBias=e.useBias==null||e.useBias,this.kernelInitializer=kb(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=kb(e.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=kb(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelRegularizer=MC(e.kernelRegularizer),this.recurrentRegularizer=MC(e.recurrentRegularizer),this.biasRegularizer=MC(e.biasRegularizer),this.kernelConstraint=_x(e.kernelConstraint),this.recurrentConstraint=_x(e.recurrentConstraint),this.biasConstraint=_x(e.biasConstraint),this.dropout=jy([1,My([0,e.dropout==null?0:e.dropout])]),this.recurrentDropout=jy([1,My([0,e.recurrentDropout==null?0:e.recurrentDropout])]),this.dropoutFunc=e.dropoutFunc,this.implementation=e.implementation,this.stateSize=this.units,this.dropoutMask=null,this.recurrentDropoutMask=null}build(e){e=Mb(e);let t=e[e.length-1];this.kernel=this.addWeight(`kernel`,[t,this.units*3],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight(`recurrent_kernel`,[this.units,this.units*3],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.bias=this.useBias?this.addWeight(`bias`,[this.units*3],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):null,this.built=!0}call(e,t){return F(()=>{if(e=e,e.length!==2)throw new K(`GRUCell expects 2 input Tensors (inputs, h, c), got ${e.length}.`);let n=t.training!=null&&t.training,r=e[1];e=e[0],0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=_w({ones:()=>$u(e),rate:this.dropout,training:n,count:3,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=_w({ones:()=>$u(r),rate:this.recurrentDropout,training:n,count:3,dropoutFunc:this.dropoutFunc}));let i=this.dropoutMask,a=this.recurrentDropoutMask,o,s,c;0<this.dropout&&this.dropout<1&&(e=z(e,i[0]));let l=Yy(e,this.kernel.read());this.useBias&&(l=$y(l,this.bias.read())),0<this.recurrentDropout&&this.recurrentDropout<1&&(r=z(r,a[0]));let u=this.recurrentKernel.read(),[d,f]=hf(u,[2*this.units,this.units],u.rank-1),p=Yy(r,d),[m,h,g]=hf(l,3,l.rank-1),[_,v]=hf(p,2,p.rank-1);o=this.recurrentActivation.apply(L(m,_)),s=this.recurrentActivation.apply(L(h,v));let y=Yy(z(s,r),f);c=this.activation.apply(L(g,y));let b=L(z(o,r),z(L(1,fu(o)),c));return[b,b]})}getConfig(){let e=super.getConfig(),t={units:this.units,activation:CC(this.activation),recurrentActivation:CC(this.recurrentActivation),useBias:this.useBias,kernelInitializer:Ob(this.kernelInitializer),recurrentInitializer:Ob(this.recurrentInitializer),biasInitializer:Ob(this.biasInitializer),kernelRegularizer:AC(this.kernelRegularizer),recurrentRegularizer:AC(this.recurrentRegularizer),biasRegularizer:AC(this.biasRegularizer),activityRegularizer:AC(this.activityRegularizer),kernelConstraint:hx(this.kernelConstraint),recurrentConstraint:hx(this.recurrentConstraint),biasConstraint:hx(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout,implementation:this.implementation,resetAfter:!1};return Object.assign(Object.assign({},e),t)}};fw.className=`GRUCell`,G(fw);var pw=class extends cw{constructor(e){e.implementation===0&&console.warn("`implementation=0` has been deprecated, and now defaults to `implementation=1`. Please update your layer call."),e.cell=new fw(e),super(e)}call(e,t){return F(()=>{this.cell.dropoutMask!=null&&(ya(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(ya(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);let n=t==null?null:t.mask,r=t==null?null:t.training,i=t==null?null:t.initialState;return super.call(e,{mask:n,training:r,initialState:i})})}static fromConfig(e,t){return t.implmentation===0&&(t.implementation=1),new e(t)}};pw.className=`GRU`,G(pw);var mw=class extends lw{constructor(e){super(e),this.DEFAULT_ACTIVATION=`tanh`,this.DEFAULT_RECURRENT_ACTIVATION=`hardSigmoid`,this.DEFAULT_KERNEL_INITIALIZER=`glorotNormal`,this.DEFAULT_RECURRENT_INITIALIZER=`orthogonal`,this.DEFAULT_BIAS_INITIALIZER=`zeros`,this.units=e.units,ry(this.units,`units`),this.activation=TC(e.activation===void 0?this.DEFAULT_ACTIVATION:e.activation),this.recurrentActivation=TC(e.recurrentActivation===void 0?this.DEFAULT_RECURRENT_ACTIVATION:e.recurrentActivation),this.useBias=e.useBias==null||e.useBias,this.kernelInitializer=kb(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=kb(e.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=kb(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.unitForgetBias=e.unitForgetBias,this.kernelRegularizer=MC(e.kernelRegularizer),this.recurrentRegularizer=MC(e.recurrentRegularizer),this.biasRegularizer=MC(e.biasRegularizer),this.kernelConstraint=_x(e.kernelConstraint),this.recurrentConstraint=_x(e.recurrentConstraint),this.biasConstraint=_x(e.biasConstraint),this.dropout=jy([1,My([0,e.dropout==null?0:e.dropout])]),this.recurrentDropout=jy([1,My([0,e.recurrentDropout==null?0:e.recurrentDropout])]),this.dropoutFunc=e.dropoutFunc,this.implementation=e.implementation,this.stateSize=[this.units,this.units],this.dropoutMask=null,this.recurrentDropoutMask=null}build(e){var t;e=Mb(e);let n=e[e.length-1];this.kernel=this.addWeight(`kernel`,[n,this.units*4],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight(`recurrent_kernel`,[this.units,this.units*4],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint);let r;if(this.useBias){if(this.unitForgetBias){let e=this.biasInitializer,n=this.units;r=new(t=class extends lb{apply(t,r){let i=e.apply([n]),a=new db().apply([n]),o=e.apply([n*2]);return Ky(Ky(i,a),o)}},t.className=`CustomInit`,t)}else r=this.biasInitializer;this.bias=this.addWeight(`bias`,[this.units*4],null,r,this.biasRegularizer,!0,this.biasConstraint)}else this.bias=null;this.built=!0}call(e,t){return F(()=>{let n=t.training!=null&&t.training;if(e=e,e.length!==3)throw new K(`LSTMCell expects 3 input Tensors (inputs, h, c), got ${e.length}.`);let r=e[1],i=e[2];e=e[0],0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=_w({ones:()=>$u(e),rate:this.dropout,training:n,count:4,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=_w({ones:()=>$u(r),rate:this.recurrentDropout,training:n,count:4,dropoutFunc:this.dropoutFunc}));let a=this.dropoutMask,o=this.recurrentDropoutMask,s,c,l,u;0<this.dropout&&this.dropout<1&&(e=z(e,a[0]));let d=Yy(e,this.kernel.read());0<this.recurrentDropout&&this.recurrentDropout<1&&(r=z(r,o[0])),d=L(d,Yy(r,this.recurrentKernel.read())),this.useBias&&(d=$y(d,this.bias.read()));let[f,p,m,h]=hf(d,4,d.rank-1);s=this.recurrentActivation.apply(f),c=this.recurrentActivation.apply(p),l=L(z(c,i),z(s,this.activation.apply(m))),u=this.recurrentActivation.apply(h);let g=z(u,this.activation.apply(l));return[g,g,l]})}getConfig(){let e=super.getConfig(),t={units:this.units,activation:CC(this.activation),recurrentActivation:CC(this.recurrentActivation),useBias:this.useBias,kernelInitializer:Ob(this.kernelInitializer),recurrentInitializer:Ob(this.recurrentInitializer),biasInitializer:Ob(this.biasInitializer),unitForgetBias:this.unitForgetBias,kernelRegularizer:AC(this.kernelRegularizer),recurrentRegularizer:AC(this.recurrentRegularizer),biasRegularizer:AC(this.biasRegularizer),activityRegularizer:AC(this.activityRegularizer),kernelConstraint:hx(this.kernelConstraint),recurrentConstraint:hx(this.recurrentConstraint),biasConstraint:hx(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout,implementation:this.implementation};return Object.assign(Object.assign({},e),t)}};mw.className=`LSTMCell`,G(mw);var hw=class extends cw{constructor(e){e.implementation===0&&console.warn("`implementation=0` has been deprecated, and now defaults to `implementation=1`. Please update your layer call."),e.cell=new mw(e),super(e)}call(e,t){return F(()=>{this.cell.dropoutMask!=null&&(ya(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(ya(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);let n=t==null?null:t.mask,r=t==null?null:t.training,i=t==null?null:t.initialState;return super.call(e,{mask:n,training:r,initialState:i})})}static fromConfig(e,t){return t.implmentation===0&&(t.implementation=1),new e(t)}};hw.className=`LSTM`,G(hw);var gw=class extends lw{constructor(e){super(e),this.cells=e.cells}get stateSize(){let e=[];for(let t of this.cells.slice().reverse())Array.isArray(t.stateSize)?e.push(...t.stateSize):e.push(t.stateSize);return e}call(e,t){return F(()=>{e=e;let n=e.slice(1),r=[];for(let e of this.cells.slice().reverse())Array.isArray(e.stateSize)?r.push(n.splice(0,e.stateSize.length)):r.push(n.splice(0,1));r.reverse();let i=[],a;for(let o=0;o<this.cells.length;++o){let s=this.cells[o];n=r[o],a=o===0?[e[0]].concat(n):[a[0]].concat(n),a=s.call(a,t),i.push(a.slice(1))}n=[];for(let e of i.slice().reverse())n.push(...e);return[a[0]].concat(n)})}build(e){Ab(e)&&(e=e[0]),e=e;let t;this.cells.forEach((n,r)=>{Cy(`RNNCell_${r}`,()=>{n.build(e),t=Array.isArray(n.stateSize)?n.stateSize[0]:n.stateSize,e=[e[0],t]})}),this.built=!0}getConfig(){let e=super.getConfig(),t={cells:this.cells.map(e=>({className:e.getClassName(),config:e.getConfig()}))};return Object.assign(Object.assign({},e),t)}static fromConfig(e,t,n={}){let r=[];for(let e of t.cells)r.push(kx(e,n));return new e({cells:r})}get trainableWeights(){if(!this.trainable)return[];let e=[];for(let t of this.cells)e.push(...t.trainableWeights);return e}get nonTrainableWeights(){let e=[];for(let t of this.cells)e.push(...t.nonTrainableWeights);if(!this.trainable){let t=[];for(let e of this.cells)t.push(...e.trainableWeights);return t.concat(e)}return e}getWeights(){let e=[];for(let t of this.cells)e.push(...t.weights);return Lb(e)}setWeights(e){let t=[];for(let n of this.cells){let r=n.weights.length,i=e.splice(r);for(let e=0;e<n.weights.length;++e)t.push([n.weights[e],i[e]])}Rb(t)}};gw.className=`StackedRNNCells`,G(gw);function _w(e){let{ones:t,rate:n,training:r=!1,count:i=1,dropoutFunc:a}=e,o=()=>a==null?nb(t(),n):a(t(),n),s=()=>ib(o,t,r);return!i||i<=1?ba(s().clone()):Array(i).fill(void 0).map(s).map(e=>ba(e.clone()))}var vw=function(e,t){var n={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&t.indexOf(r)<0&&(n[r]=e[r]);if(e!=null&&typeof Object.getOwnPropertySymbols==`function`)for(var i=0,r=Object.getOwnPropertySymbols(e);i<r.length;i++)t.indexOf(r[i])<0&&Object.prototype.propertyIsEnumerable.call(e,r[i])&&(n[r[i]]=e[r[i]]);return n},yw=class extends cw{constructor(e){if(e.unroll)throw new Lv(`Unrolling is not possible with convolutional RNNs.`);if(Array.isArray(e.cell))throw new Lv(`It is not possible at the moment to stack convolutional cells.`);super(e),this.inputSpec=[new zb({ndim:5})]}call(e,t){return F(()=>{if(this.cell.dropoutMask!=null&&(ya(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(ya(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null),t&&t.constants)throw new K(`ConvRNN2D cell does not support constants`);let n=t==null?null:t.mask,r=t==null?null:t.training,i=t==null?null:t.initialState;return super.call(e,{mask:n,training:r,initialState:i})})}computeOutputShape(e){let t=this.computeSingleOutputShape(e);return this.returnSequences||(t=[t[0],...t.slice(2)]),this.returnState&&(t=[t,...[,,].fill([e[0],...t.slice(-3)])]),t}getInitialState(e){return F(()=>{let{stateSize:t}=this.cell,n=e.shape,r=this.computeSingleOutputShape(n),i=Ru([r[0],...r.slice(2)]);return Array.isArray(t)?Array(t.length).fill(i):[i]})}resetStates(e,t=!1){F(()=>{if(!this.stateful)throw new Fv(`Cannot call resetStates() on an RNN Layer that is not stateful.`);let n=this.inputSpec[0].shape,r=this.computeSingleOutputShape(n),i=[r[0],...r.slice(2)];if(n[0]==null)throw new K("If an RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batchInputShape` option to your first layer.\n- If using the functional API, specify the batch size by passing a `batchShape` option to your Input layer.");if(this.getStates()==null)this.states_=Array.isArray(this.cell.stateSize)?this.cell.stateSize.map(()=>Ru(i)):[Ru(i)];else if(e==null)ya(this.states_),this.keptStates!=null&&(ya(this.keptStates),this.keptStates=[]),Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(()=>Ru(i)):this.states_[0]=Ru(i);else{if(Array.isArray(e)||(e=[e]),e.length!==this.states_.length)throw new K(`Layer ${this.name} expects ${this.states_.length} state(s), but it received ${e.length} state value(s). Input received: ${e}`);t?this.keptStates.push(this.states_.slice()):ya(this.states_);for(let t=0;t<this.states_.length;++t){let n=e[t],r=i;if(!w(n.shape,r))throw new K(`State ${t} is incompatible with layer ${this.name}: expected shape=${r}, received shape=${n.shape}`);this.states_[t]=n}}this.states_=this.states_.map(e=>ba(e.clone()))})}computeSingleOutputShape(e){let{dataFormat:t,filters:n,kernelSize:r,padding:i,strides:a,dilationRate:o}=this.cell,s=t===`channelsFirst`,c=e[s?3:2],l=e[s?4:3],u=BC(c,r[0],i,a[0],o[0]),d=BC(l,r[1],i,a[1],o[1]);return[...e.slice(0,2),...s?[n,u,d]:[u,d,n]]}};yw.className=`ConvRNN2D`;var bw=class extends mw{constructor(e){let{filters:t,kernelSize:n,strides:r,padding:i,dataFormat:a,dilationRate:o}=e;super(Object.assign(Object.assign({},e),{units:t})),this.filters=t,ry(this.filters,`filters`),this.kernelSize=zC(n,2,`kernelSize`),this.kernelSize.forEach(e=>ry(e,`kernelSize`)),this.strides=zC(r||1,2,`strides`),this.strides.forEach(e=>ry(e,`strides`)),this.padding=i||`valid`,yy(this.padding),this.dataFormat=a||`channelsLast`,_y(this.dataFormat),this.dilationRate=zC(o||1,2,`dilationRate`),this.dilationRate.forEach(e=>ry(e,`dilationRate`))}build(e){var t;e=Mb(e);let n=this.dataFormat===`channelsFirst`?1:e.length-1;if(e[n]==null)throw new K(`The channel dimension of the input should be defined. Found ${e[n]}`);let r=e[n],i=this.kernelSize.concat([r,this.filters*4]);this.kernel=this.addWeight(`kernel`,i,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint);let a=this.kernelSize.concat([this.filters,this.filters*4]);if(this.recurrentKernel=this.addWeight(`recurrent_kernel`,a,null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias){let e;if(this.unitForgetBias){let n=this.biasInitializer,r=this.filters;e=new(t=class extends lb{apply(e,t){return Gy([n.apply([r]),zu([r]),n.apply([r*2])])}},t.className=`CustomInit`,t)}else e=this.biasInitializer;this.bias=this.addWeight(`bias`,[this.filters*4],null,e,this.biasRegularizer,!0,this.biasConstraint)}this.built=!0}call(e,t){return F(()=>{if(e.length!==3)throw new K(`ConvLSTM2DCell expects 3 input Tensors (inputs, h, c), got ${e.length}.`);let n=t.training||!1,r=e[0],i=e[1],a=e[2];0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=_w({ones:()=>$u(r),rate:this.dropout,training:n,count:4,dropoutFunc:this.dropoutFunc}));let o=this.dropoutMask,s=(e,t,n)=>!t||!t[n]?e:z(t[n],e),c=s(r,o,0),l=s(r,o,1),u=s(r,o,2),d=s(r,o,3);0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=_w({ones:()=>$u(i),rate:this.recurrentDropout,training:n,count:4,dropoutFunc:this.dropoutFunc}));let f=this.recurrentDropoutMask,p=s(i,f,0),m=s(i,f,1),h=s(i,f,2),g=s(i,f,3),[_,v,y,b]=hf(this.kernel.read(),4,3),[x,S,C,w]=this.useBias?hf(this.bias.read(),4):[null,null,null,null];c=this.inputConv(c,_,x,this.padding),l=this.inputConv(l,v,S,this.padding),u=this.inputConv(u,y,C,this.padding),d=this.inputConv(d,b,w,this.padding);let[T,E,D,ee]=hf(this.recurrentKernel.read(),4,3);p=this.recurrentConv(p,T),m=this.recurrentConv(m,E),h=this.recurrentConv(h,D),g=this.recurrentConv(g,ee);let te=this.recurrentActivation.apply(L(c,p)),O=L(z(this.recurrentActivation.apply(L(l,m)),a),z(te,this.activation.apply(L(u,h)))),ne=z(this.recurrentActivation.apply(L(d,g)),this.activation.apply(O));return[ne,ne,O]})}getConfig(){let e=super.getConfig(),{units:t}=e,n=vw(e,[`units`]),r={filters:this.filters,kernelSize:this.kernelSize,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,strides:this.strides};return Object.assign(Object.assign({},n),r)}inputConv(e,t,n,r){let i=cc(e,t,this.strides,r||`valid`,this.dataFormat===`channelsFirst`?`NCHW`:`NHWC`,this.dilationRate);return n?$y(i,n,this.dataFormat):i}recurrentConv(e,t){return cc(e,t,1,`same`,this.dataFormat===`channelsFirst`?`NCHW`:`NHWC`)}};bw.className=`ConvLSTM2DCell`,G(bw);var xw=class extends yw{constructor(e){let t=new bw(e);super(Object.assign(Object.assign({},e),{cell:t}))}static fromConfig(e,t){return new e(t)}};xw.className=`ConvLSTM2D`,G(xw);var Sw=class extends Wb{constructor(e){super(e),this.rate=Math.max(Math.min(e.rate,1),0),this.noiseShape=e.noiseShape,this.seed=e.seed,this.supportsMasking=!0}getNoiseShape(e){if(this.noiseShape==null)return this.noiseShape;let t=e.shape,n=[];for(let e=0;e<this.noiseShape.length;++e)n.push(this.noiseShape[e]==null?t[e]:this.noiseShape[e]);return n}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e);if(0<this.rate&&this.rate<1){let e=t.training!=null&&t.training,r=this.getNoiseShape(n);return ib(()=>nb(n,this.rate,r,this.seed),()=>n,e)}return e})}getConfig(){let e={rate:this.rate,noiseShape:this.noiseShape,seed:this.seed},t=super.getConfig();return Object.assign(e,t),e}dispose(){return super.dispose()}};Sw.className=`Dropout`,G(Sw);var Cw=class extends Sw{constructor(e){super(e),this.inputSpec=[{ndim:3}]}getNoiseShape(e){let t=e.shape;return[t[0],1,t[2]]}};Cw.className=`SpatialDropout1D`,G(Cw);var ww=class extends Wb{constructor(e){if(super(e),this.activation=null,this.useBias=!0,this.kernel=null,this.bias=null,this.DEFAULT_KERNEL_INITIALIZER=`glorotNormal`,this.DEFAULT_BIAS_INITIALIZER=`zeros`,e.batchInputShape==null&&e.inputShape==null&&e.inputDim!=null){let t=null;e.batchSize!=null&&(t=e.batchSize),this.batchInputShape=[t,e.inputDim]}this.units=e.units,ry(this.units,`units`),this.activation=TC(e.activation),e.useBias!=null&&(this.useBias=e.useBias),this.kernelInitializer=kb(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.biasInitializer=kb(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelConstraint=_x(e.kernelConstraint),this.biasConstraint=_x(e.biasConstraint),this.kernelRegularizer=MC(e.kernelRegularizer),this.biasRegularizer=MC(e.biasRegularizer),this.activityRegularizer=MC(e.activityRegularizer),this.supportsMasking=!0,this.inputSpec=[{minNDim:2}]}build(e){e=Mb(e);let t=e[e.length-1];this.kernel??(this.kernel=this.addWeight(`kernel`,[t,this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight(`bias`,[this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint))),this.inputSpec=[{minNDim:2,axes:{[-1]:t}}],this.built=!0}computeOutputShape(e){e=Mb(e);let t=e.slice();return t[t.length-1]=this.units,t}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e),r=oy(this.activation.getClassName()),i;return r==null?(i=Yy(n,this.kernel.read()),this.bias!=null&&(i=$y(i,this.bias.read())),this.activation!=null&&(i=this.activation.apply(i))):i=Yy(n,this.kernel.read(),r,this.bias?this.bias.read():null),i})}getConfig(){let e={units:this.units,activation:CC(this.activation),useBias:this.useBias,kernelInitializer:Ob(this.kernelInitializer),biasInitializer:Ob(this.biasInitializer),kernelRegularizer:AC(this.kernelRegularizer),biasRegularizer:AC(this.biasRegularizer),activityRegularizer:AC(this.activityRegularizer),kernelConstraint:hx(this.kernelConstraint),biasConstraint:hx(this.biasConstraint)},t=super.getConfig();return Object.assign(e,t),e}};ww.className=`Dense`,G(ww);var Tw=class extends Wb{constructor(e){e||={},super(e),this.inputSpec=[{minNDim:3}],this.dataFormat=e.dataFormat}computeOutputShape(e){e=Mb(e);for(let t of e.slice(1))if(t==null)throw new K(`The shape of the input to "Flatten" is not fully defined (got ${e.slice(1)}). Make sure to pass a complete "input_shape" or "batch_input_shape" argument to the first layer in your model.`);return[e[0],Ay(e,1)]}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e);if(this.dataFormat===`channelsFirst`&&n.rank>1){let e=[0];for(let t=2;t<n.rank;++t)e.push(t);e.push(1),n=Yf(n,e)}return Vy(n)})}getConfig(){let e={};this.dataFormat!=null&&(e.dataFormat=this.dataFormat);let t=super.getConfig();return Object.assign(e,t),e}};Tw.className=`Flatten`,G(Tw);var Ew=class extends Wb{constructor(e){super(e),this.supportsMasking=!0,this.activation=TC(e.activation)}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e);return this.activation.apply(n)})}getConfig(){let e={activation:CC(this.activation)},t=super.getConfig();return Object.assign(e,t),e}};Ew.className=`Activation`,G(Ew);var Dw=class extends Wb{constructor(e){super(e),this.n=e.n,this.inputSpec=[{ndim:2}]}computeOutputShape(e){return[e[0],this.n,e[1]]}call(e,t){return F(()=>(e=q(e),zy(e,this.n)))}getConfig(){let e={n:this.n},t=super.getConfig();return Object.assign(e,t),e}};Dw.className=`RepeatVector`,G(Dw);var Ow=class extends Wb{constructor(e){super(e),this.targetShape=e.targetShape;for(let e=0;e<this.targetShape.length;++e)this.isUnknown(this.targetShape[e])&&(this.targetShape[e]=null)}isUnknown(e){return e<0||e==null}fixUnknownDimension(e,t){let n=`Total size of new array must be unchanged.`,r=t.slice(),i=1,a=null;for(let e=0;e<r.length;++e){let t=r[e];if(this.isUnknown(t)){if(a===null)a=e;else throw new K(`Can only specifiy one unknown dimension.`)}else i*=t}let o=Ay(e);if(a!==null){if(i===0||o%i!==0)throw new K(n);r[a]=o/i}else if(o!==i)throw new K(n);return r}computeOutputShape(e){let t=!1;for(let n=0;n<e.length;++n)if(this.isUnknown(e[n])){t=!0;break}return t?e.slice(0,1).concat(this.targetShape):e.slice(0,1).concat(this.fixUnknownDimension(e.slice(1),this.targetShape))}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e),r=n.shape;return B(n,r.slice(0,1).concat(this.fixUnknownDimension(r.slice(1),this.targetShape)))})}getConfig(){let e={targetShape:this.targetShape},t=super.getConfig();return Object.assign(e,t),e}};Ow.className=`Reshape`,G(Ow);var kw=class extends Wb{constructor(e){if(super(e),e.dims==null)throw Error("Required configuration field `dims` is missing during Permute constructor call.");if(!Array.isArray(e.dims))throw Error(`Permute constructor requires \`dims\` to be an Array, but received ${e.dims} instead.`);let t=Ny(1,e.dims.length+1);if(!w(e.dims.slice().sort(),t))throw Error("Invalid permutation `dims`: "+JSON.stringify(e.dims)+" `dims` must contain consecutive integers starting from 1.");this.dims=e.dims,this.dimsIncludingBatch=[0].concat(this.dims),this.inputSpec=[new zb({ndim:this.dims.length+1})]}computeOutputShape(e){e=Mb(e);let t=e.slice();return this.dims.forEach((n,r)=>{t[r+1]=e[n]}),t}call(e,t){return Yf(q(e),this.dimsIncludingBatch)}getConfig(){let e={dims:this.dims},t=super.getConfig();return Object.assign(e,t),e}};kw.className=`Permute`,G(kw);var Aw=class extends Wb{constructor(e){super(e??{}),this.supportsMasking=!0,this.maskValue=e==null||e.maskValue==null?0:e.maskValue}computeOutputShape(e){return e}getConfig(){let e=super.getConfig(),t={maskValue:this.maskValue};return Object.assign(t,e),t}computeMask(e,t){return Ro(Yu(q(e),this.maskValue),-1)}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e);return z(n,I(Ro(Yu(n,this.maskValue),-1,!0),n.dtype))})}};Aw.className=`Masking`,G(Aw);var jw=class extends Wb{constructor(e){if(super(e),this.embeddings=null,this.DEFAULT_EMBEDDINGS_INITIALIZER=`randomUniform`,e.batchInputShape==null&&e.inputShape==null){let t=null;e.batchSize!=null&&(t=e.batchSize),this.batchInputShape=e.inputLength==null?[t,null]:[t].concat(Wv(e.inputLength))}this.inputDim=e.inputDim,ry(this.inputDim,`inputDim`),this.outputDim=e.outputDim,ry(this.outputDim,`outputDim`),this.embeddingsInitializer=kb(e.embeddingsInitializer||this.DEFAULT_EMBEDDINGS_INITIALIZER),this.embeddingsRegularizer=MC(e.embeddingsRegularizer),this.activityRegularizer=MC(e.activityRegularizer),this.embeddingsConstraint=_x(e.embeddingsConstraint),this.maskZero=e.maskZero,this.supportsMasking=e.maskZero,this.inputLength=e.inputLength}build(e){this.embeddings=this.addWeight(`embeddings`,[this.inputDim,this.outputDim],this.dtype,this.embeddingsInitializer,this.embeddingsRegularizer,!0,this.embeddingsConstraint),this.built=!0}warnOnIncompatibleInputShape(e){}computeMask(e,t){return F(()=>this.maskZero?(e=q(e),Yu(e,Wc(e))):null)}computeOutputShape(e){if(e=Mb(e),this.inputLength==null)return[...e,this.outputDim];let t=Wv(this.inputLength);if(t.length!==e.length-1)throw new K(`"inputLength" is ${this.inputLength}, but received input shape has shape ${e}`);{let n=0;for(let r=0;r<t.length;++r){let i=t[r],a=e[r+1];if(i!=null&&a!=null&&i!==a)throw new K(`"inputLength" is ${this.inputLength}, but received input shape has shape ${e}`);i??(t[n]=a),n++}}return[e[0],...t,this.outputDim]}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e);return n.dtype!==`int32`&&(n=Ly(n,`int32`)),B(Xy(this.embeddings.read(),B(n,[n.size])),Mb(this.computeOutputShape(n.shape)))})}getConfig(){let e={inputDim:this.inputDim,outputDim:this.outputDim,embeddingsInitializer:Ob(this.embeddingsInitializer),embeddingsRegularizer:AC(this.embeddingsRegularizer),activityRegularizer:AC(this.activityRegularizer),embeddingsConstraint:hx(this.embeddingsConstraint),maskZero:this.maskZero,inputLength:this.inputLength},t=super.getConfig();return Object.assign(e,t),e}};jw.className=`Embedding`,G(jw);var Mw=class extends Wb{constructor(e){super(e||{}),this.supportsMasking=!0}mergeFunction(e){throw new Lv}computeElementwiseOpOutputShape(e,t){if(e==null||t==null)return null;if(e.length<t.length)return this.computeElementwiseOpOutputShape(t,e);if(t.length===0)return e;let n=e.slice(0,e.length-t.length);for(let r=0;r<t.length;++r){let i=e[e.length-t.length+r],a=t[r];if(i==null||a==null||i<0||a<0)n.push(null);else if(i===1)n.push(a);else if(a===1)n.push(i);else{if(i!==a)throw new K(`Operands could not be broadcast together with shapes `+JSON.stringify(e)+` `+JSON.stringify(t));n.push(i)}}return n}build(e){if(Array.isArray(e)&&!Array.isArray(e[0])&&(e=[Mb(e)]),e=e,e.length<2)throw new K(`A merge layer should be called on an Array of at least 2 inputs. Got ${e.length} input(s).`);let t=[];for(let n of e)n!=null&&n[0]!==null&&t.push(n[0]);if(t=$v(t),t.length>1)throw new K(`Can not merge tensors with different batch sizes. Got tensors with shapes: ${JSON.stringify(e)}.`);let n=e[0]==null?null:e[0].slice(1);for(let t=1;t<e.length;++t){let r=e[t]==null?null:e[t].slice(1);n=this.computeElementwiseOpOutputShape(n,r)}let r=e.map(e=>e.length);this.reshapeRequired=e.indexOf(null)!==-1||$v(r).length!==1}call(e,t){return F(()=>{if(e=e,this.reshapeRequired){let t=[],n=e.map(e=>e.rank);if(n.indexOf(null)===-1){let r=My(n);for(let n of e){let e=n.rank;for(let t=0;t<r-e;++t)n=Ry(n,1);t.push(n)}return this.mergeFunction(t)}{let n=!1;for(let r of e){let e=r.rank;if(e==null){let e=r.shape,i=e[0],a=e.slice(1).concat([i]),o=B(r,[i].concat(Ay(e.slice(1))));o=Yf(o,[1,0]),o=B(o,a),t.push(o),n=!0}else if(e>1){let i=Ny(1,e).concat([0]);t.push(Yf(r,i)),n=!0}else t.push(r)}let r=this.mergeFunction(t),i=r.rank;if(n){if(i==null){let e=r.shape,t=e[e.length-1],n=[t].concat(e.slice(0,e.length-1));r=B(Yf(B(r,[-1,t]),[1,0]),n)}else if(i>1){let e=[i-1].concat(Ny(0,i-1));r=Yf(r,e)}}return r}}return this.mergeFunction(e)})}computeOutputShape(e){e=e;let t;t=e[0]==null?null:e[0].slice(1);for(let n=1;n<e.length;++n){let r=e[n]==null?null:e[n].slice(1);t=this.computeElementwiseOpOutputShape(t,r)}let n=[];for(let t of e)t!=null&&t[0]!==null&&n.push(t[0]);return n=$v(n),t=n.length===1?n.concat(t):[null].concat(t),t}computeMask(e,t){return F(()=>{if(t==null)return null;if(!Array.isArray(t))throw new K("`mask` should be an Array");if(!Array.isArray(e))throw new K("`inputs` should be an Array");if(t.length!==e.length)throw new K(`The Array 'inputs' and 'mask' are expected to have the same length, but have different lengths (${e.length} vs ${t.length})`);if(t.every(e=>e==null))return null;t=t.map(e=>e==null?e:Ol(e,0));let n=t[0];for(let e=1;e<t.length-1;++e)n=Cu(n,t[e]);return n})}},Nw=class extends Mw{constructor(e){super(e)}mergeFunction(e){return F(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=L(t,e[n]);return t})}};Nw.className=`Add`,G(Nw);var Pw=class extends Mw{constructor(e){super(e)}mergeFunction(e){return F(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=z(t,e[n]);return t})}};Pw.className=`Multiply`,G(Pw);var Fw=class extends Mw{constructor(e){super(e)}mergeFunction(e){return F(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=L(t,e[n]);return z(1/e.length,t)})}};Fw.className=`Average`,G(Fw);var Iw=class extends Mw{constructor(e){super(e)}mergeFunction(e){return F(()=>{let t=e[0];for(let n=1;n<e.length;++n)t=Fu(t,e[n]);return t})}};Iw.className=`Maximum`,G(Iw);var Lw=class extends Mw{constructor(e){super(e)}mergeFunction(e){return F(()=>{let t=e[0];for(let n=1;n<e.length;++n)t=Vu(t,e[n]);return t})}};Lw.className=`Minimum`,G(Lw);var Rw=class extends Mw{constructor(e){super(e),this.DEFAULT_AXIS=-1,e??={},this.axis=e.axis==null?this.DEFAULT_AXIS:e.axis,this.supportsMasking=!0,this.reshapeRequired=!1}build(e){if(!(Array.isArray(e)&&Array.isArray(e[0]))||e.length===1)throw new K("A `Concatenate` layer should be called on a list of at least 2 inputs");e=e;let t=!0;for(let n of e)if(n!=null){t=!1;break}if(t)return;let n=[];for(let t=0;t<e.length;++t){let r=e[t].slice();r.splice(this.axis,1);let i=!1;for(let e of n)if(w(e,r)){i=!0;break}i||n.push(r)}if(n.length>1)throw new K("A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got input shapes: "+JSON.stringify(e))}mergeFunction(e){return F(()=>Gy(e,this.axis))}computeOutputShape(e){if(!(Array.isArray(e)&&Array.isArray(e[0])))throw new K("A `Concatenate` layer should be called on a list of inputs.");let t=e,n=t[0].slice(),r=this.axis<0?n.length+this.axis:this.axis;for(let e of t.slice(1)){if(n[r]==null||e[r]==null){n[r]=null;break}n[r]+=e[r]}return n}computeMask(e,t){if(t==null)return null;if(!Array.isArray(t))throw new K("`mask` should be an array for Concatenate");if(!Array.isArray(e))throw new K("`inputs` should be an array for Concatenate");if(t.length!==e.length)throw new K(`Mismatch in the length of mask (${t.length}) and the legnth of inputs (${e.length})`);return F(()=>{let n=!0;if(t.forEach(e=>{if(e!=null){n=!1;return}}),n)return null;let r=[];for(let n=0;n<e.length;++n)t[n]==null?r.push(I($u(e[n]),`bool`)):t[n].rank<e[n].rank?r.push(Ol(t[n],-1)):r.push(t[n]);return Io(ws(r,this.axis),-1,!1)})}getConfig(){let e={axis:this.axis},t=super.getConfig();return Object.assign(e,t),e}};Rw.className=`Concatenate`,G(Rw);function zw(e,t){for(;e<0;)e+=t;return e}function Bw(e,t,n){if(e.shape.length>3||t.shape.length>3)throw new Lv(`batchDot is not implemented for tensors of 4D or higher rank yet`);if(b(e.shape.length>=2,()=>`batchDot requires the rank of x to be >= 2, but got ${e.shape.length}`),b(e.shape.length>=2,()=>`batchDot requires the rank of y to be >= 2, but got ${t.shape.length}`),typeof n==`number`&&(n=[n,n]),e.dtype===`complex64`||t.dtype===`complex64`)throw new Lv(`batchDot is not implemented for complex64-type Tensors yet.`);let r=e.shape.length,i=t.shape.length;n??=[r-1,i-2];let a=n;return F(()=>{let n;if(r>i){n=r-i;let e=[];for(let t=0;t<n;++t)e.push(1);t=B(t,t.shape.concat(e))}else if(i>r){n=i-r;let t=[];for(let e=0;e<n;++e)t.push(1);e=B(e,e.shape.concat(t))}else n=0;let o;if(e.shape.length===2&&t.shape.length===2)o=a[0]===a[1]?U(z(e,t),a[0]):U(z(Yf(e,[1,0]),t),a[1]);else{let n=a[0]!==e.shape.length-1,r=a[1]===t.shape.length-1;o=Es(e,t,n,r)}if(n>0){let e;e=r>i?r+i-3:r-1;let t=[];for(let r=e;r<e+n;++r)t.push(r);o=xf(o,t)}return o.shape.length===1&&(o=Ol(o,1)),o})}var Vw=class extends Mw{constructor(e){super(e),this.axes=e.axes,this.normalize=e.normalize!=null&&e.normalize,this.supportsMasking=!0,this.reshapeRequired=!1}build(e){b(Array.isArray(e)&&e.length===2&&Array.isArray(e[0])&&Array.isArray(e[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");let t=e[0],n=e[1];if(t.length>3||n.length>3)throw new Lv(`Dot layer does not support tensors of 4D or higher rank yet.`);let r=this.interpretAxes(t,n);if(t[r[0]]!==n[r[1]])throw new K(`Dimension incompatibility: ${t[r[0]]} !== ${n[r[1]]}`)}mergeFunction(e){if(e.length!==2)throw new K(`A \`Dot\` layer must be called on exactly 2 inputs, but received ${e.length} input(s).`);let t=e[0],n=e[1],r;return r=Array.isArray(this.axes)?this.axes.map((t,n)=>zw(t,e[n].shape.length)):[zw(this.axes,t.shape.length),zw(this.axes,n.shape.length)],this.normalize&&(t=Ax(t,r[0]),n=Ax(n,r[1])),Bw(t,n,r)}interpretAxes(e,t){let n;return n=Array.isArray(this.axes)?this.axes:[zw(this.axes,e.length),zw(this.axes,t.length)],n}computeOutputShape(e){b(Array.isArray(e)&&e.length===2&&Array.isArray(e[0])&&Array.isArray(e[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");let t=e[0].slice(),n=e[1].slice();if(t.length>3||n.length>3)throw new Lv(`Dot layer does not support tensors of 4D or higher rank yet.`);let r=this.interpretAxes(t,n);t.splice(r[0],1),n.splice(r[1],1),n.splice(0,1);let i=t.concat(n);return i.length===1&&i.push(1),i}computeMask(e,t){return null}getConfig(){let e={axes:this.axes,normalize:this.normalize},t=super.getConfig();return Object.assign(e,t),e}};Vw.className=`Dot`,G(Vw);var Hw=class extends Wb{constructor(e){super(e),this.supportsMasking=!0,this.stddev=e.stddev}computeOutputShape(e){return e}getConfig(){let e=super.getConfig(),t={stddev:this.stddev};return Object.assign(t,e),t}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e);return ib(()=>L(Jy(n.shape,0,this.stddev),n),()=>n,t.training||!1)})}};Hw.className=`GaussianNoise`,G(Hw);var Uw=class extends Wb{constructor(e){super(e),this.supportsMasking=!0,this.rate=e.rate}computeOutputShape(e){return e}getConfig(){let e=super.getConfig(),t={rate:this.rate};return Object.assign(t,e),t}call(e,t){return F(()=>{this.invokeCallHook(e,t);let n=q(e);return this.rate>0&&this.rate<1?ib(()=>{let e=Math.sqrt(this.rate/(1-this.rate));return z(n,Jy(n.shape,1,e))},()=>n,t.training||!1):n})}};Uw.className=`GaussianDropout`,G(Uw);var Ww=class extends Wb{constructor(e){super(e),this.supportsMasking=!0,this.rate=e.rate,this.noiseShape=e.noiseShape}_getNoiseShape(e){return this.noiseShape||q(e).shape}computeOutputShape(e){return e}getConfig(){let e=super.getConfig(),t={rate:this.rate};return Object.assign(t,e),t}call(e,t){return F(()=>{if(this.rate<1&&this.rate>0){let n=this._getNoiseShape(e);return ib(()=>{let t=q(e),r=-1.7580993408473766,i=Hl(Td(n),this.rate);i=Ly(i,`float32`);let a=((1-this.rate)*(1+this.rate*r**2))**-.5,o=-a*r*this.rate;return L(z(L(z(t,i),z(L(i,-1),r)),a),o)},()=>q(e),t.training||!1)}return e})}};Ww.className=`AlphaDropout`,G(Ww);function Gw(e,t,n,r,i,a=.001){let o;if(e.rank===2)o=zs(e,t,n,r,i,a);else if(e.rank===3)o=Vs(e,t,n,r,i,a);else if(e.rank===4)o=Us(e,t,n,r,i,a);else throw new Lv(`batchNormalization is not implemented for array of rank ${e.rank} yet`);return o}function Kw(e,t,n,r,i=.001){return F(()=>{let a=qu(e,r),o=a.mean,s=a.variance;return[Gw(e,o,s,n,t,i),o,s]})}function qw(e,t,n,r,i=.001){return F(()=>{let a=qu(e,r),o=a.mean,s=a.variance,c=[];for(let t of Ny(0,e.rank))r.indexOf(t)===-1?c.push(e.shape[t]):c.push(1);let l=B(o,c),u=B(s,c),d=t==null?null:B(t,c);return[Gw(e,l,u,n==null?null:B(n,c),d,i),o,s]})}function Jw(e,t,n,r,i=.001){return w(r.slice().sort(),Ny(0,e.rank-1))?Kw(e,t,n,r,i):qw(e,t,n,r,i)}var Yw=class extends Wb{constructor(e){e??={},super(e),this.supportsMasking=!0,this.axis=e.axis==null?-1:e.axis,this.momentum=e.momentum==null?.99:e.momentum,this.epsilon=e.epsilon==null?.001:e.epsilon,this.center=e.center==null||e.center,this.scale=e.scale==null||e.scale,this.betaInitializer=kb(e.betaInitializer||`zeros`),this.gammaInitializer=kb(e.gammaInitializer||`ones`),this.movingMeanInitializer=kb(e.movingMeanInitializer||`zeros`),this.movingVarianceInitializer=kb(e.movingVarianceInitializer||`ones`),this.betaConstraint=_x(e.betaConstraint),this.gammaConstraint=_x(e.gammaConstraint),this.betaRegularizer=MC(e.betaRegularizer),this.gammaRegularizer=MC(e.gammaRegularizer)}build(e){e=Mb(e);let t=this.axis>=0?this.axis:this.axis+e.length,n=e[t];if(n==null)throw new K(`Axis ${t} of input tensor should have a defined dimension but the layer received an input with shape ${JSON.stringify(e)}.`);this.inputSpec=[new zb({ndim:e.length,axes:{[t]:n}})];let r=[n];this.scale&&(this.gamma=this.addWeight(`gamma`,r,null,this.gammaInitializer,this.gammaRegularizer,!0,this.gammaConstraint)),this.center&&(this.beta=this.addWeight(`beta`,r,null,this.betaInitializer,this.betaRegularizer,!0,this.betaConstraint)),this.movingMean=this.addWeight(`moving_mean`,r,null,this.movingMeanInitializer,null,!1),this.movingVariance=this.addWeight(`moving_variance`,r,null,this.movingVarianceInitializer,null,!1),this.built=!0}call(e,t){return F(()=>{let n=t.training!=null&&t.training,r=q(e),i=r.shape,a=i.length,o=Ny(0,a),s=this.axis>=0?this.axis:this.axis+a;o.splice(s,1);let c=Bv(1,a);c[s]=i[s];let l=o.slice();l.sort();let u=!w(l,Ny(0,a).slice(0,a-1)),d=()=>{if(u){let e=B(this.movingMean.read(),c),t=B(this.movingVariance.read(),c),n=this.center?B(this.beta.read(),c):null,i=this.scale?B(this.gamma.read(),c):null;return Gw(r,e,t,n,i,this.epsilon)}return Gw(r,this.movingMean.read(),this.movingVariance.read(),this.beta==null?null:this.beta.read(),this.gamma==null?null:this.gamma.read(),this.epsilon)};if(!n)return d();let[f,p,m]=Jw(r,this.gamma.read(),this.beta.read(),o,this.epsilon),h=(e,t,n)=>{F(()=>{let r=1-n,i=e.read(),a=z(W(i,t),r);e.write(W(i,a))})};return h(this.movingMean,p,this.momentum),h(this.movingVariance,m,this.momentum),f})}getConfig(){let e={axis:this.axis,momentum:this.momentum,epsilon:this.epsilon,center:this.center,scale:this.scale,betaInitializer:Ob(this.betaInitializer),gammaInitializer:Ob(this.gammaInitializer),movingMeanInitializer:Ob(this.movingMeanInitializer),movingVarianceInitializer:Ob(this.movingVarianceInitializer),betaRegularizer:AC(this.betaRegularizer),gammaRegularizer:AC(this.gammaRegularizer),betaConstraint:hx(this.betaConstraint),gammaConstraint:hx(this.gammaConstraint)},t=super.getConfig();return Object.assign(e,t),e}};Yw.className=`BatchNormalization`,G(Yw);var Xw=class extends Wb{constructor(e){if(e??={},super(e),this.axis=e.axis==null?-1:e.axis,typeof this.axis==`number`){if(!Number.isInteger(this.axis))throw Error(`Expected axis to be an integer, but received ${this.axis}`)}else if(Array.isArray(this.axis)){for(let e of this.axis)if(!Number.isInteger(e))throw Error(`Expected axis to be an array of integers, but received ${JSON.stringify(this.axis)}`)}else throw Error(`Expected axis to be an integer or an array of integers, but received ${JSON.stringify(this.axis)}`);this.epsilon=e.epsilon==null?.001:e.epsilon,this.center=e.center==null||e.center,this.scale=e.scale==null||e.scale,this.betaInitializer=kb(e.betaInitializer||`zeros`),this.gammaInitializer=kb(e.gammaInitializer||`ones`),this.betaRegularizer=MC(e.betaRegularizer),this.gammaRegularizer=MC(e.gammaRegularizer),this.supportsMasking=!0}build(e){e=Mb(e);let t=e.length;typeof this.axis==`number`&&(this.axis=[this.axis]);for(let e=0;e<this.axis.length;++e)this.axis[e]<0&&(this.axis[e]+=t);for(let e of this.axis)if(e<0||e>=t)throw Error(`Invalid axis: ${e}`);if(this.axis.length!==$v(this.axis).length)throw Error(`Found duplicate axes in: ${this.axis}`);let n=this.axis.map(t=>e[t]);this.gamma=this.scale?this.addWeight(`gamma`,n,`float32`,this.gammaInitializer,this.gammaRegularizer,!0):null,this.beta=this.center?this.addWeight(`beta`,n,`float32`,this.betaInitializer,this.betaRegularizer,!0):null,this.built=!0}call(e,t){let n=q(e),r=n.shape,i=r.length;return F(()=>{let{mean:e,variance:t}=qu(n,this.axis,!0),a=Bv(1,i);for(let e of this.axis)a[e]=r[e];let o=e=>e!=null&&e.shape.length!==i?B(e,a):e,s=this.scale?o(this.gamma.read()):null,c=this.center?o(this.beta.read()):null,l=[],u=[];for(let e=0;e<i;++e)this.axis.indexOf(e)===-1?(l.push(1),u.push(r[e])):(l.push(r[e]),u.push(1));return e=Ml(e,l),t=Ml(t,l),s!=null&&(s=Ml(s,u)),c!=null&&(c=Ml(c,u)),Gw(n,e,t,c,s,this.epsilon)})}getConfig(){let e={axis:this.axis,epsilon:this.epsilon,center:this.center,scale:this.scale,betaInitializer:Ob(this.betaInitializer),gammaInitializer:Ob(this.gammaInitializer),betaRegularizer:AC(this.betaRegularizer),gammaRegularizer:AC(this.gammaRegularizer)},t=super.getConfig();return Object.assign(e,t),e}};Xw.className=`LayerNormalization`,G(Xw);function Zw(e,t,n){return F(()=>{if(e.rank!==4)throw new K(`temporalPadding expects input tensor to be 4-D, but received a ${e.rank}-D tensor.`);if(t??=[[1,1],[1,1]],t.length!==2||t[0].length!==2||t[1].length!==2)throw new K("spatial2dPadding expects `padding` to be an Array of two Arrays, each of which is an Array of two integers.");if(n??=Iy(),n!==`channelsLast`&&n!==`channelsFirst`)throw new K(`Unknown data format: ${n}. Supported data formats are 'channelsLast' and 'channelsFirst.`);let r;return r=n===`channelsFirst`?[[0,0],[0,0],t[0],t[1]]:[[0,0],t[0],t[1],[0,0]],td(e,r)})}var Qw=class extends Wb{constructor(e){if(e??={},super(e),this.dataFormat=e.dataFormat==null?Iy():e.dataFormat,e.padding==null)this.padding=[[1,1],[1,1]];else if(typeof e.padding==`number`)this.padding=[[e.padding,e.padding],[e.padding,e.padding]];else{if(e.padding=e.padding,e.padding.length!==2)throw new K(`ZeroPadding2D expects padding to be a length-2 array, but received a length-${e.padding.length} array.`);let t,n;if(typeof e.padding[0]==`number`)t=[e.padding[0],e.padding[0]],n=[e.padding[1],e.padding[1]];else{if(e.padding=e.padding,e.padding[0].length!==2)throw new K(`ZeroPadding2D expects height padding to be a length-2 array, but received a length-${e.padding[0].length} array.`);if(t=e.padding[0],e.padding[1].length!==2)throw new K(`ZeroPadding2D expects width padding to be a length-2 array, but received a length-${e.padding[1].length} array.`);n=e.padding[1]}this.padding=[t,n]}this.inputSpec=[new zb({ndim:4})]}computeOutputShape(e){e=Mb(e);let t,n;return this.dataFormat===`channelsFirst`?(t=e[2]!=null&&e[2]>=0?e[2]+this.padding[0][0]+this.padding[0][1]:null,n=e[3]!=null&&e[3]>=0?e[3]+this.padding[1][0]+this.padding[1][1]:null,[e[0],e[1],t,n]):(t=e[1]!=null&&e[1]>=0?e[1]+this.padding[0][0]+this.padding[0][1]:null,n=e[2]!=null&&e[2]>=0?e[2]+this.padding[1][0]+this.padding[1][1]:null,[e[0],t,n,e[3]])}call(e,t){return F(()=>Zw(q(e),this.padding,this.dataFormat))}getConfig(){let e={padding:this.padding,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}};Qw.className=`ZeroPadding2D`,G(Qw);function $w(e,t,n,r,i,a){return F(()=>{_y(i),by(a),yy(r),n??=[1,1],r??=`valid`,i??=Iy(),a??=`max`,e=HC(e,i);let o,s=r===`same`?`same`:`valid`;return o=a===`max`?ju(e,t,n,s):bs(e,t,n,s),i===`channelsFirst`&&(o=Yf(o,[0,3,1,2])),o})}function eT(e,t,n,r,i,a){return F(()=>{_y(i),by(a),yy(r),n??=[1,1,1],r??=`valid`,i??=Iy(),a??=`max`,e=UC(e,i);let o,s=r===`same`?`same`:`valid`;return o=a===`max`?Nu(e,t,n,s):Ss(e,t,n,s),i===`channelsFirst`&&(o=Yf(o,[0,4,1,2,3])),o})}var tT=class extends Wb{constructor(e){if(e.poolSize??=2,super(e),typeof e.poolSize==`number`)this.poolSize=[e.poolSize];else if(Array.isArray(e.poolSize)&&e.poolSize.length===1&&typeof e.poolSize[0]==`number`)this.poolSize=e.poolSize;else throw new K(`poolSize for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(e.poolSize)}`);if(ry(this.poolSize,`poolSize`),e.strides==null)this.strides=this.poolSize;else if(typeof e.strides==`number`)this.strides=[e.strides];else if(Array.isArray(e.strides)&&e.strides.length===1&&typeof e.strides[0]==`number`)this.strides=e.strides;else throw new K(`strides for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(e.strides)}`);ry(this.strides,`strides`),this.padding=e.padding==null?`valid`:e.padding,yy(this.padding),this.inputSpec=[new zb({ndim:3})]}computeOutputShape(e){e=Mb(e);let t=BC(e[1],this.poolSize[0],this.padding,this.strides[0]);return[e[0],t,e[2]]}call(e,t){return F(()=>(this.invokeCallHook(e,t),e=Ry(q(e),2),xf(this.poolingFunction(q(e),[this.poolSize[0],1],[this.strides[0],1],this.padding,`channelsLast`),[2])))}getConfig(){let e={poolSize:this.poolSize,padding:this.padding,strides:this.strides},t=super.getConfig();return Object.assign(e,t),e}},nT=class extends tT{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return _y(i),yy(r),$w(e,t,n,r,i,`max`)}};nT.className=`MaxPooling1D`,G(nT);var rT=class extends tT{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return _y(i),yy(r),$w(e,t,n,r,i,`avg`)}};rT.className=`AveragePooling1D`,G(rT);var iT=class extends Wb{constructor(e){if(e.poolSize??=[2,2],super(e),this.poolSize=Array.isArray(e.poolSize)?e.poolSize:[e.poolSize,e.poolSize],e.strides==null)this.strides=this.poolSize;else if(Array.isArray(e.strides)){if(e.strides.length!==2)throw new K(`If the strides property of a 2D pooling layer is an Array, it is expected to have a length of 2, but received length ${e.strides.length}.`);this.strides=e.strides}else this.strides=[e.strides,e.strides];ry(this.poolSize,`poolSize`),ry(this.strides,`strides`),this.padding=e.padding==null?`valid`:e.padding,this.dataFormat=e.dataFormat==null?`channelsLast`:e.dataFormat,_y(this.dataFormat),yy(this.padding),this.inputSpec=[new zb({ndim:4})]}computeOutputShape(e){e=Mb(e);let t=this.dataFormat===`channelsFirst`?e[2]:e[1],n=this.dataFormat===`channelsFirst`?e[3]:e[2];return t=BC(t,this.poolSize[0],this.padding,this.strides[0]),n=BC(n,this.poolSize[1],this.padding,this.strides[1]),this.dataFormat===`channelsFirst`?[e[0],e[1],t,n]:[e[0],t,n,e[3]]}call(e,t){return F(()=>(this.invokeCallHook(e,t),this.poolingFunction(q(e),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){let e={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}},aT=class extends iT{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return _y(i),yy(r),$w(e,t,n,r,i,`max`)}};aT.className=`MaxPooling2D`,G(aT);var oT=class extends iT{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return _y(i),yy(r),$w(e,t,n,r,i,`avg`)}};oT.className=`AveragePooling2D`,G(oT);var sT=class extends Wb{constructor(e){if(e.poolSize??=[2,2,2],super(e),this.poolSize=Array.isArray(e.poolSize)?e.poolSize:[e.poolSize,e.poolSize,e.poolSize],e.strides==null)this.strides=this.poolSize;else if(Array.isArray(e.strides)){if(e.strides.length!==3)throw new K(`If the strides property of a 3D pooling layer is an Array, it is expected to have a length of 3, but received length ${e.strides.length}.`);this.strides=e.strides}else this.strides=[e.strides,e.strides,e.strides];ry(this.poolSize,`poolSize`),ry(this.strides,`strides`),this.padding=e.padding==null?`valid`:e.padding,this.dataFormat=e.dataFormat==null?`channelsLast`:e.dataFormat,_y(this.dataFormat),yy(this.padding),this.inputSpec=[new zb({ndim:5})]}computeOutputShape(e){e=Mb(e);let t=this.dataFormat===`channelsFirst`?e[2]:e[1],n=this.dataFormat===`channelsFirst`?e[3]:e[2],r=this.dataFormat===`channelsFirst`?e[4]:e[3];return t=BC(t,this.poolSize[0],this.padding,this.strides[0]),n=BC(n,this.poolSize[1],this.padding,this.strides[1]),r=BC(r,this.poolSize[2],this.padding,this.strides[2]),this.dataFormat===`channelsFirst`?[e[0],e[1],t,n,r]:[e[0],t,n,r,e[4]]}call(e,t){return F(()=>(this.invokeCallHook(e,t),this.poolingFunction(q(e),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){let e={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}},cT=class extends sT{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return _y(i),yy(r),eT(e,t,n,r,i,`max`)}};cT.className=`MaxPooling3D`,G(cT);var lT=class extends sT{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return _y(i),yy(r),eT(e,t,n,r,i,`avg`)}};lT.className=`AveragePooling3D`,G(lT);var uT=class extends Wb{constructor(e){super(e),this.inputSpec=[new zb({ndim:3})]}computeOutputShape(e){return[e[0],e[2]]}call(e,t){throw new Lv}},dT=class extends uT{constructor(e){super(e||{})}call(e,t){return F(()=>Lu(q(e),1))}};dT.className=`GlobalAveragePooling1D`,G(dT);var fT=class extends uT{constructor(e){super(e||{})}call(e,t){return F(()=>ul(q(e),1))}};fT.className=`GlobalMaxPooling1D`,G(fT);var pT=class extends Wb{constructor(e){super(e),this.dataFormat=e.dataFormat==null?`channelsLast`:e.dataFormat,_y(this.dataFormat),this.inputSpec=[new zb({ndim:4})]}computeOutputShape(e){return e=e,this.dataFormat===`channelsLast`?[e[0],e[3]]:[e[0],e[1]]}call(e,t){throw new Lv}getConfig(){let e={dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}},mT=class extends pT{call(e,t){return F(()=>{let t=q(e);return this.dataFormat===`channelsLast`?Lu(t,[1,2]):Lu(t,[2,3])})}};mT.className=`GlobalAveragePooling2D`,G(mT);var hT=class extends pT{call(e,t){return F(()=>{let t=q(e);return this.dataFormat===`channelsLast`?ul(t,[1,2]):ul(t,[2,3])})}};hT.className=`GlobalMaxPooling2D`,G(hT);var gT=class extends Wb{constructor(e){super(e),this.layer=e.layer}build(e){this.built=!0}get trainable(){return this.layer!=null&&this.layer.trainable}set trainable(e){this.layer!=null&&(this.layer.trainable=e)}get trainableWeights(){return this.layer.trainableWeights}get nonTrainableWeights(){return this.layer.nonTrainableWeights}get updates(){return this.layer._updates}get losses(){return this.layer.losses}getWeights(){return this.layer.getWeights()}setWeights(e){this.layer.setWeights(e)}getConfig(){let e={layer:{className:this.layer.getClassName(),config:this.layer.getConfig()}},t=super.getConfig();return Object.assign(e,t),e}setFastWeightInitDuringBuild(e){super.setFastWeightInitDuringBuild(e),this.layer!=null&&this.layer.setFastWeightInitDuringBuild(e)}static fromConfig(e,t,n={}){let r=t.layer,i=kx(r,n);delete t.layer;let a={layer:i};return Object.assign(a,t),new e(a)}},_T=class extends gT{constructor(e){super(e),this.supportsMasking=!0}build(e){if(e=Mb(e),e.length<3)throw new K(`TimeDistributed layer expects an input shape >= 3D, but received input shape ${JSON.stringify(e)}`);this.inputSpec=[{shape:e}];let t=[e[0]].concat(e.slice(2));this.layer.built||(this.layer.build(t),this.layer.built=!0),super.build(e)}computeOutputShape(e){e=Mb(e);let t=[e[0]].concat(e.slice(2)),n=this.layer.computeOutputShape(t),r=e[1];return[n[0],r].concat(n.slice(1))}call(e,t){return F(()=>(e=q(e),sw((e,n)=>[q(this.layer.call(e,t)),[]],e,[],!1,null,null,!1,!0)[1]))}};_T.className=`TimeDistributed`,G(_T);function vT(e){ty(hy,`BidirectionalMergeMode`,e)}var yT=`concat`,bT=class extends gT{constructor(e){super(e);let t=e.layer.getConfig(),n={};n.className=e.layer.getClassName(),n.config=t,this.forwardLayer=kx(n),t.goBackwards=t.goBackwards!==!0;let r={};if(r.className=e.layer.getClassName(),r.config=t,this.backwardLayer=kx(r),this.forwardLayer.name=`forward_`+this.forwardLayer.name,this.backwardLayer.name=`backward_`+this.backwardLayer.name,this.mergeMode=e.mergeMode===void 0?yT:e.mergeMode,vT(this.mergeMode),e.weights)throw new Lv(`weights support is not implemented for Bidirectional layer yet.`);this._stateful=e.layer.stateful,this.returnSequences=e.layer.returnSequences,this.returnState=e.layer.returnState,this.supportsMasking=!0,this._trainable=!0,this.inputSpec=e.layer.inputSpec,this.numConstants=null}get trainable(){return this._trainable}set trainable(e){this._trainable=e,this.forwardLayer!=null&&(this.forwardLayer.trainable=e),this.backwardLayer!=null&&(this.backwardLayer.trainable=e)}getWeights(){return this.forwardLayer.getWeights().concat(this.backwardLayer.getWeights())}setWeights(e){let t=e.length,n=Math.floor(t/2);this.forwardLayer.setWeights(e.slice(0,n)),this.backwardLayer.setWeights(e.slice(n))}computeOutputShape(e){let t=this.forwardLayer.computeOutputShape(e);Array.isArray(t)&&Array.isArray(t[0])||(t=[t]),t=t;let n,r,i;return this.returnState&&(i=t.slice(1)),n=t[0],n=n,this.mergeMode===`concat`?(n[n.length-1]*=2,r=[n]):r=this.mergeMode==null?[n,n.slice()]:[n],this.returnState?this.mergeMode==null?r.concat(i).concat(i.slice()):[n].concat(i,i.slice()):Uv(r)}apply(e,t){let n=t==null?null:t.initialState,r=t==null?null:t.constants;t??={};let i=ow(e,n,r,this.numConstants);if(e=i.inputs,n=i.initialState,r=i.constants,Array.isArray(e)&&(n=e.slice(1),e=e[0]),(n==null||n.length===0)&&r==null)return super.apply(e,t);let a=[],o=[];if(n!=null){let e=n.length;if(e%2>0)throw new K("When passing `initialState` to a Bidrectional RNN, the state should be an Array containing the states of the underlying RNNs.");t.initialState=n,a.push(...n);let r=n.map(e=>new zb({shape:e.shape}));this.forwardLayer.stateSpec=r.slice(0,e/2),this.backwardLayer.stateSpec=r.slice(e/2),o.push(...r)}if(r!=null)throw new Lv(`Support for constants in Bidirectional layers is not implemented yet.`);let s=a[0]instanceof Bb;for(let e of a)if(e instanceof Bb!==s)throw new K(`The initial state of a Bidirectional layer cannot be specified as a mix of symbolic and non-symbolic tensors`);if(s){let n=[e].concat(a),r=this.inputSpec.concat(o),i=this.inputSpec;this.inputSpec=r;let s=super.apply(n,t);return this.inputSpec=i,s}return super.apply(e,t)}call(e,t){return F(()=>{let n=t.initialState,r,i;if(n==null)r=this.forwardLayer.call(e,t),i=this.backwardLayer.call(e,t);else{let a=n.slice(0,n.length/2),o=n.slice(n.length/2);r=this.forwardLayer.call(e,Object.assign(t,{initialState:a})),i=this.backwardLayer.call(e,Object.assign(t,{initialState:o}))}let a;this.returnState&&(Array.isArray(r)&&(a=r.slice(1).concat(i.slice(1))),r=r[0],i=i[0]),this.returnSequences&&(i=Id(i,1));let o;return this.mergeMode===`concat`?o=Gy([r,i]):this.mergeMode===`sum`?o=L(r,i):this.mergeMode===`ave`?o=z(.5,L(r,i)):this.mergeMode===`mul`?o=z(r,i):this.mergeMode??(o=[r,i]),this.returnState?this.mergeMode==null?o.concat(a):[o].concat(a):o})}resetStates(e){this.forwardLayer.resetStates(),this.backwardLayer.resetStates()}build(e){Cy(this.forwardLayer.name,()=>{this.forwardLayer.build(e)}),Cy(this.backwardLayer.name,()=>{this.backwardLayer.build(e)}),this.built=!0}computeMask(e,t){Array.isArray(t)&&(t=t[0]);let n;if(n=this.returnSequences?this.mergeMode==null?[t,t]:t:this.mergeMode==null?[null,null]:null,this.returnState){let e=this.forwardLayer.states.map(e=>null);return Array.isArray(n)?n.concat(e).concat(e):[n].concat(e,e)}return n}get trainableWeights(){return this.forwardLayer.trainableWeights.concat(this.backwardLayer.trainableWeights)}get nonTrainableWeights(){return this.forwardLayer.nonTrainableWeights.concat(this.backwardLayer.nonTrainableWeights)}setFastWeightInitDuringBuild(e){super.setFastWeightInitDuringBuild(e),this.forwardLayer!=null&&this.forwardLayer.setFastWeightInitDuringBuild(e),this.backwardLayer!=null&&this.backwardLayer.setFastWeightInitDuringBuild(e)}getConfig(){let e={mergeMode:this.mergeMode},t=super.getConfig();return Object.assign(e,t),e}static fromConfig(e,t){let n=kx(t.layer);if(delete t.layer,t.numConstants!=null)throw new Lv(`Deserialization of a Bidirectional layer with numConstants present is not supported yet.`);let r=t;return r.layer=n,new e(r)}};bT.className=`Bidirectional`,G(bT);var xT=class extends Wb{constructor(e){super(e),this.scale=e.scale,this.offset=e.offset?e.offset:0}getConfig(){let e={scale:this.scale,offset:this.offset},t=super.getConfig();return Object.assign(e,t),e}call(e,t){return F(()=>(e=q(e),e.dtype!==`float32`&&(e=Ly(e,`float32`)),L(z(e,this.scale),this.offset)))}};xT.className=`Rescaling`,G(xT);var{resizeBilinear:ST,cropAndResize:CT}=Am,wT=class extends Wb{constructor(e){super(e),this.height=e.height,this.width=e.width}centerCrop(e,t,n,r,i,a,o,s){return F(()=>{let c,l=!1,u=[t/a,n/o,(r+t)/a,(i+n)/o],d=[];e.rank===3?(l=!0,c=Cf([e])):c=e;for(let e=0;e<c.shape[0];e++)d.push(u);let f=ma(d,[d.length,4]),p=Ed(0,d.length,1,`int32`),m=CT(c,f,p,[r,i],`nearest`);return Ly(l?q(Gf(m)):m,s)})}upsize(e,t,n,r){return F(()=>Ly(ST(e,[t,n]),r))}call(e,t){return F(()=>{let t=q(e),n=t.dtype,r=t.shape,i=r[r.length-3],a=r[r.length-2],o=0;i!==this.height&&(o=Math.floor((i-this.height)/2));let s=0;return a!==this.width&&(s=Math.floor((a-this.width)/2),s===0&&(s=1)),o>=0&&s>=0?this.centerCrop(t,o,s,this.height,this.width,i,a,n):this.upsize(e,this.height,this.width,n)})}getConfig(){let e={height:this.height,width:this.width},t=super.getConfig();return Object.assign(e,t),e}computeOutputShape(e){e=Mb(e);let t=e.length-3,n=e.length-2;return e[t]=this.height,e[n]=this.width,e}};wT.className=`CenterCrop`,G(wT);function TT(e,t,n,r){let i=q(e);if(i.dtype!==`int32`&&(i=Ly(i,`int32`)),t===`int`)return i;let a=i.shape;if(i.rank===0&&(i=Ol(i,-1)),t===`oneHot`&&i.shape[i.shape.length-1]!==1&&(i=Ol(i,-1)),i.rank>2)throw new K(`When outputMode is not int, maximum output rank is 2 Received outputMode ${t} and input shape ${a} which would result in output rank ${i.rank}.`);let o=[`multiHot`,`oneHot`].includes(t),s=i,c;if(c=r!==void 0&&t===`count`?Ac(s,r,n,o):Ac(s,[],n,o),t!==`tfIdf`)return c;if(r)return z(c,r);throw new K(`When outputMode is 'tfIdf', weights must be provided.`)}var ET=class extends Wb{constructor(e){super(e),this.numTokens=e.numTokens,this.outputMode=e.outputMode?e.outputMode:`multiHot`}getConfig(){let e={numTokens:this.numTokens,outputMode:this.outputMode},t=super.getConfig();return Object.assign(e,t),e}computeOutputShape(e){return e=Mb(e),e==null?[this.numTokens]:this.outputMode===`oneHot`&&e[e.length-1]!==1?(e.push(this.numTokens),e):(e[e.length-1]=this.numTokens,e)}call(e,t){return F(()=>{e=q(e),e.dtype!==`int32`&&(e=Ly(e,`int32`));let n;if(t.countWeights!==void 0){if(this.outputMode!==`count`)throw new K(`countWeights is not used when outputMode !== count.
              Received countWeights=${t.countWeights}`);n=q(t.countWeights)}let r=ul(e),i=fl(e),a=Bl(this.numTokens,r).bufferSync().get(0),o=Hl(i,0).bufferSync().get(0);if(!(a&&o))throw new K(`Input values must be between 0 < values <= numTokens with numTokens=${this.numTokens}`);return TT(e,this.outputMode,this.numTokens,n)})}};ET.className=`CategoryEncoding`,G(ET);var DT=new Set([`bilinear`,`nearest`]),OT=class extends Wb{constructor(e){if(super(e),this.height=e.height,this.width=e.width,e.interpolation){if(DT.has(e.interpolation))this.interpolation=e.interpolation;else throw new K(`Invalid interpolation parameter: ${e.interpolation} is not implemented`)}else this.interpolation=`bilinear`;this.cropToAspectRatio=!!e.cropToAspectRatio}computeOutputShape(e){e=Mb(e);let t=e[2];return[this.height,this.width,t]}getConfig(){let e={height:this.height,width:this.width,interpolation:this.interpolation,cropToAspectRatio:this.cropToAspectRatio},t=super.getConfig();return Object.assign(e,t),e}call(e,t){return F(()=>{let t=[this.height,this.width];if(this.interpolation===`bilinear`)return Am.resizeBilinear(e,t,!this.cropToAspectRatio);if(this.interpolation===`nearest`)return Am.resizeNearestNeighbor(e,t,!this.cropToAspectRatio);throw Error(`Interpolation is ${this.interpolation} but only ${[...DT]} are supported`)})}};OT.className=`Resizing`,G(OT);var kT=class{constructor(e){this.seed=e}next(){if(this.seed!==void 0)return this.seed++}};kT.className=`RandomSeed`;var AT=class extends Wb{constructor(e){super(e),this.randomGenerator=new kT(e.seed)}getConfig(){let e={seed:this.randomGenerator.seed},t=super.getConfig();return Object.assign(e,t),e}};AT.className=`BaseRandomLayer`;var jT=new Set([`bilinear`,`nearest`]),MT=class extends AT{constructor(e){super(e);let{factor:t,interpolation:n=`bilinear`}=e;if(this.factor=t,Array.isArray(this.factor)&&this.factor.length===2)this.widthLower=this.factor[0],this.widthUpper=this.factor[1];else if(!Array.isArray(this.factor)&&this.factor>0)this.widthLower=-this.factor,this.widthUpper=this.factor;else throw new K(`Invalid factor: ${this.factor}. Must be positive number or tuple of 2 numbers`);if(this.widthLower<-1||this.widthUpper<-1)throw new K(`factor must have values larger than -1. Got: ${this.factor}`);if(this.widthUpper<this.widthLower)throw new K(`factor cannot have upper bound less than lower bound.
        Got upper bound: ${this.widthUpper}.
        Got lower bound: ${this.widthLower}
      `);if(n){if(jT.has(n))this.interpolation=n;else throw new K(`Invalid interpolation parameter: ${n} is not implemented`)}}getConfig(){let e={factor:this.factor,interpolation:this.interpolation},t=super.getConfig();return Object.assign(e,t),e}computeOutputShape(e){e=Mb(e);let t=e[2];return[this.imgHeight,-1,t]}call(e,t){return F(()=>{let t=q(e);this.imgHeight=t.shape[t.shape.length-3];let n=t.shape[t.shape.length-2];this.widthFactor=Td([1],1+this.widthLower,1+this.widthUpper,`float32`,this.randomGenerator.next());let r=this.widthFactor.dataSync()[0]*n;r=Math.round(r);let i=[this.imgHeight,r];switch(this.interpolation){case`bilinear`:return Am.resizeBilinear(e,i);case`nearest`:return Am.resizeNearestNeighbor(e,i);default:throw Error(`Interpolation is ${this.interpolation}
          but only ${[...jT]} are supported`)}})}};MT.className=`RandomWidth`,G(MT),A().registerFlag(`KEEP_INTERMEDIATE_TENSORS`,()=>!1,e=>{e&&console.warn(`Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.`)});var NT;(function(e){e[e.DT_INVALID=0]=`DT_INVALID`,e[e.DT_FLOAT=1]=`DT_FLOAT`,e[e.DT_DOUBLE=2]=`DT_DOUBLE`,e[e.DT_INT32=3]=`DT_INT32`,e[e.DT_UINT8=4]=`DT_UINT8`,e[e.DT_INT16=5]=`DT_INT16`,e[e.DT_INT8=6]=`DT_INT8`,e[e.DT_STRING=7]=`DT_STRING`,e[e.DT_COMPLEX64=8]=`DT_COMPLEX64`,e[e.DT_INT64=9]=`DT_INT64`,e[e.DT_BOOL=10]=`DT_BOOL`,e[e.DT_QINT8=11]=`DT_QINT8`,e[e.DT_QUINT8=12]=`DT_QUINT8`,e[e.DT_QINT32=13]=`DT_QINT32`,e[e.DT_BFLOAT16=14]=`DT_BFLOAT16`,e[e.DT_QINT16=15]=`DT_QINT16`,e[e.DT_QUINT16=16]=`DT_QUINT16`,e[e.DT_UINT16=17]=`DT_UINT16`,e[e.DT_COMPLEX128=18]=`DT_COMPLEX128`,e[e.DT_HALF=19]=`DT_HALF`,e[e.DT_RESOURCE=20]=`DT_RESOURCE`,e[e.DT_VARIANT=21]=`DT_VARIANT`,e[e.DT_UINT32=22]=`DT_UINT32`,e[e.DT_UINT64=23]=`DT_UINT64`,e[e.DT_FLOAT_REF=101]=`DT_FLOAT_REF`,e[e.DT_DOUBLE_REF=102]=`DT_DOUBLE_REF`,e[e.DT_INT32_REF=103]=`DT_INT32_REF`,e[e.DT_UINT8_REF=104]=`DT_UINT8_REF`,e[e.DT_INT16_REF=105]=`DT_INT16_REF`,e[e.DT_INT8_REF=106]=`DT_INT8_REF`,e[e.DT_STRING_REF=107]=`DT_STRING_REF`,e[e.DT_COMPLEX64_REF=108]=`DT_COMPLEX64_REF`,e[e.DT_INT64_REF=109]=`DT_INT64_REF`,e[e.DT_BOOL_REF=110]=`DT_BOOL_REF`,e[e.DT_QINT8_REF=111]=`DT_QINT8_REF`,e[e.DT_QUINT8_REF=112]=`DT_QUINT8_REF`,e[e.DT_QINT32_REF=113]=`DT_QINT32_REF`,e[e.DT_BFLOAT16_REF=114]=`DT_BFLOAT16_REF`,e[e.DT_QINT16_REF=115]=`DT_QINT16_REF`,e[e.DT_QUINT16_REF=116]=`DT_QUINT16_REF`,e[e.DT_UINT16_REF=117]=`DT_UINT16_REF`,e[e.DT_COMPLEX128_REF=118]=`DT_COMPLEX128_REF`,e[e.DT_HALF_REF=119]=`DT_HALF_REF`,e[e.DT_RESOURCE_REF=120]=`DT_RESOURCE_REF`,e[e.DT_VARIANT_REF=121]=`DT_VARIANT_REF`,e[e.DT_UINT32_REF=122]=`DT_UINT32_REF`,e[e.DT_UINT64_REF=123]=`DT_UINT64_REF`})(NT||={});var PT;(function(e){(function(e){e[e.LEGACY=0]=`LEGACY`,e[e.V1=1]=`V1`,e[e.V2=2]=`V2`})(e.CheckpointFormatVersion||={})})(PT||={});function FT(e,t){return IT(e,t)}function IT(e,t,n=new Map,r=new Set){if(e==null)return null;if(typeof Blob==`function`&&e instanceof Blob)return e.slice();if(r.has(e))throw Error(`Circular references are not supported.`);if(n.has(e))return n.get(e);let i=t(e);if(i.recurse&&i.value!==null)throw Error(`A deep map function may not return both a value and recurse=true.`);if(!i.recurse)return n.set(e,i.value),i.value;if(BT(e)){let i=Array.isArray(e)?[]:{};r.add(e);for(let a in e){let o=e[a];i[a]=IT(o,t,n,r)}return r.delete(e),e.__proto__&&(i.__proto__=e.__proto__),i}throw Error(`Can't recurse into non-iterable type: ${e}`)}function LT(e,t=zT){return RT(e,t)}function RT(e,t,n=new Set){let r=e[0];if(n.has(r))throw Error(`Circular references are not supported.`);let i=t(e);if(i.recurse&&i.value!==null)throw Error(`A deep zip function may not return both a value and recurse=true.`);if(!i.recurse)return i.value;if(BT(r)){let i=Array.isArray(r)?[]:{};n.add(r);for(let a in r)i[a]=RT(e.map(e=>e[a]),t,n);return n.delete(r),i}throw Error(`Can't recurse into non-iterable type: ${r}`)}function zT(e){return e===null?null:BT(e[0])?{value:null,recurse:!0}:{value:e,recurse:!1}}function BT(e){let t=!1;if(A().get(`IS_BROWSER`))t=e instanceof TextDecoder;else{let{StringDecoder:n}=po();t=e instanceof n}return e!=null&&!ArrayBuffer.isView(e)&&(Array.isArray(e)||typeof e==`object`&&!(e instanceof Ni)&&!(e instanceof Promise)&&!t)}function VT(e){return e==null||HT(e)||Array.isArray(e)||typeof e==`object`&&e instanceof Ni||fi(e)}function HT(e){return e===null||typeof e!=`object`&&typeof e!=`function`}function UT(e){return FT(e,WT)}function WT(e){return e instanceof Ni?{value:e.clone(),recurse:!1}:BT(e)?{value:null,recurse:!0}:{value:e,recurse:!1}}var GT=class{constructor(e){if(this.capacity=e,this.begin=0,this.end=0,e==null)throw RangeError(`Can't create a ring buffer of unknown capacity.`);if(e<1)throw RangeError(`Can't create ring buffer of capacity < 1.`);this.data=Array(e),this.doubledCapacity=2*e}wrap(e){for(;e<0;)e+=this.doubledCapacity;return e%this.doubledCapacity}get(e){if(e<0)throw RangeError(`Can't get item at a negative index.`);return this.data[e%this.capacity]}set(e,t){if(e<0)throw RangeError(`Can't set item at a negative index.`);this.data[e%this.capacity]=t}length(){let e=this.end-this.begin;return e<0&&(e=this.doubledCapacity+e),e}isFull(){return this.length()===this.capacity}isEmpty(){return this.length()===0}push(e){if(this.isFull())throw RangeError(`Ring buffer is full.`);this.set(this.end,e),this.end=this.wrap(this.end+1)}pushAll(e){for(let t of e)this.push(t)}pop(){if(this.isEmpty())throw RangeError(`Ring buffer is empty.`);this.end=this.wrap(this.end-1);let e=this.get(this.end);return this.set(this.end,void 0),e}unshift(e){if(this.isFull())throw RangeError(`Ring buffer is full.`);this.begin=this.wrap(this.begin-1),this.set(this.begin,e)}shift(){if(this.isEmpty())throw RangeError(`Ring buffer is empty.`);let e=this.get(this.begin);return this.set(this.begin,void 0),this.begin=this.wrap(this.begin+1),e}shuffleExcise(e){if(this.isEmpty())throw RangeError(`Ring buffer is empty.`);let t=this.wrap(this.begin+e),n=this.get(t);return this.set(t,this.pop()),n}},KT=class e extends GT{constructor(){super(e.INITIAL_CAPACITY)}isFull(){return!1}push(e){super.isFull()&&this.expand(),super.push(e)}unshift(e){super.isFull()&&this.expand(),super.unshift(e)}expand(){let e=this.capacity*2,t=Array(e),n=this.length();for(let e=0;e<n;e++)t[e]=this.get(this.wrap(this.begin+e));this.data=t,this.capacity=e,this.doubledCapacity=2*this.capacity,this.begin=0,this.end=n}};KT.INITIAL_CAPACITY=32;function qT(e){return new ZT(e)}function JT(e){return new QT(e)}function YT(e,t){return new lE(e,t)}var XT=class{async toArray(){let e=[],t=await this.next();for(;!t.done;)e.push(t.value),t=await this.next();return e}async toArrayForTest(){let e=this.prefetch(100),t=[],n=await e.next();for(;!n.done;)t.push(n.value),n=await e.next();return t}async resolveFully(){let e=await this.next();for(;!e.done;)e=await this.next()}async resolveWhile(e){let t=await this.next(),n=e(t.value);for(;!t.done&&n;)t=await this.next(),n=e(t.value)}handleErrors(e){return new aE(this,e)}filter(e){return new rE(this,e)}map(e){return new iE(this,e)}mapAsync(e){return new oE(this,e)}serialMapAsync(e){return new oE(this,e).serial()}flatmap(e){return new cE(this,e)}async forEachAsync(e){return this.map(e).resolveFully()}async serialForEach(e){return this.serialMapAsync(e).resolveWhile(e=>e===!0)}rowMajorBatch(e,t=!0){return new nE(this,e,t)}columnMajorBatch(e,t=!0,n=zT){return this.rowMajorBatch(e,t).map(e=>LT(e,n))}concatenate(e,t){return new lE(qT([this,e]),t)}take(e){return e<0||e==null?this:new tE(this,e)}skip(e){return e<0||e==null?this:new eE(this,e)}prefetch(e){return new dE(this,e)}shuffle(e,t){return new fE(this,e,t)}serial(){return new $T(this)}},ZT=class extends XT{constructor(e){super(),this.items=e,this.trav=0}summary(){return`Array of ${this.items.length} items`}async next(){if(this.trav>=this.items.length)return{value:null,done:!0};let e=this.items[this.trav];return this.trav++,{value:UT(e),done:!1}}},QT=class extends XT{constructor(e){super(),this.nextFn=e}summary(){return`Function call`}async next(){try{return this.nextFn()}catch(e){throw e.message=`Error thrown while iterating through a dataset: ${e.message}`,e}}},$T=class extends XT{constructor(e){super(),this.upstream=e,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Serial`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){return this.upstream.next()}},eE=class extends XT{constructor(e,t){super(),this.upstream=e,this.maxCount=t,this.count=0,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Skip`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;this.count++<this.maxCount;){let e=await this.upstream.next();if(e.done)return e;ya(e.value)}return this.upstream.next()}},tE=class extends XT{constructor(e,t){super(),this.upstream=e,this.maxCount=t,this.count=0}summary(){return`${this.upstream.summary()} -> Take`}async next(){return this.count++>=this.maxCount?{value:null,done:!0}:this.upstream.next()}},nE=class extends XT{constructor(e,t,n=!0){super(),this.upstream=e,this.batchSize=t,this.enableSmallLastBatch=n,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> RowMajorBatch`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){let e=[];for(;e.length<this.batchSize;){let t=await this.upstream.next();if(t.done)return this.enableSmallLastBatch&&e.length>0?{value:e,done:!1}:{value:null,done:!0};e.push(t.value)}return{value:e,done:!1}}},rE=class extends XT{constructor(e,t){super(),this.upstream=e,this.predicate=t,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Filter`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;;){let e=await this.upstream.next();if(e.done||this.predicate(e.value))return e;ya(e.value)}}},iE=class extends XT{constructor(e,t){super(),this.upstream=e,this.transform=t}summary(){return`${this.upstream.summary()} -> Map`}async next(){let e=await this.upstream.next();if(e.done)return{value:null,done:!0};let t=qi(e.value),n=this.transform(e.value),r=qi(n);for(let e of t)Ki(e,r)||e.dispose();return{value:n,done:!1}}},aE=class extends XT{constructor(e,t){super(),this.upstream=e,this.handler=t,this.count=0,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> handleErrors`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;;)try{return await this.upstream.next()}catch(e){if(!this.handler(e))return{value:null,done:!0}}}},oE=class extends XT{constructor(e,t){super(),this.upstream=e,this.transform=t}summary(){return`${this.upstream.summary()} -> AsyncMap`}async next(){let e=await this.upstream.next();if(e.done)return{value:null,done:!0};let t=qi(e.value),n=await this.transform(e.value),r=qi(n);for(let e of t)Ki(e,r)||e.dispose();return{value:n,done:!1}}},sE=class extends XT{constructor(){super(),this.outputQueue=new KT,this.lastRead=Promise.resolve({value:null,done:!1})}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;this.outputQueue.length()===0;)if(!await this.pump())return{value:null,done:!0};return{value:this.outputQueue.shift(),done:!1}}},cE=class extends sE{constructor(e,t){super(),this.upstream=e,this.transform=t}summary(){return`${this.upstream.summary()} -> Flatmap`}async pump(){let e=await this.upstream.next();if(e.done)return!1;let t=qi(e.value),n=this.transform(e.value),r=qi(n);this.outputQueue.pushAll(n);for(let e of t)Ki(e,r)||e.dispose();return!0}},lE=class extends XT{constructor(e,t){super(),this.baseErrorHandler=t,this.lastRead=null,this.iterator=null,this.moreIterators=e}summary(){return`TODO: fill in upstream of chained summaries -> Chained`}async next(){return this.lastRead=this.readFromChain(this.lastRead),this.lastRead}async readFromChain(e){if(await e,this.iterator==null){let e=await this.moreIterators.next();if(e.done)return{value:null,done:!0};this.iterator=e.value,this.baseErrorHandler!=null&&(this.iterator=this.iterator.handleErrors(this.baseErrorHandler))}let t=await this.iterator.next();return t.done?(this.iterator=null,this.readFromChain(e)):t}},uE;(function(e){e[e.FAIL=0]=`FAIL`,e[e.SHORTEST=1]=`SHORTEST`,e[e.LONGEST=2]=`LONGEST`})(uE||={});var dE=class extends XT{constructor(e,t){super(),this.upstream=e,this.bufferSize=t,this.buffer=new GT(t)}summary(){return`${this.upstream.summary()} -> Prefetch`}refill(){for(;!this.buffer.isFull();){let e=this.upstream.next();this.buffer.push(e)}}next(){return this.refill(),this.buffer.shift()}},fE=class extends dE{constructor(e,t,n){super(e,t),this.upstream=e,this.windowSize=t,this.upstreamExhausted=!1,this.random=yd.alea(n||li().toString()),this.lastRead=Promise.resolve({value:null,done:!1})}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}randomInt(e){return Math.floor(this.random()*e)}chooseIndex(){return this.randomInt(this.buffer.length())}async serialNext(){for(this.upstreamExhausted||this.refill();!this.buffer.isEmpty();){let e=this.chooseIndex(),t=await this.buffer.shuffleExcise(e);if(t.done)this.upstreamExhausted=!0;else return this.refill(),t}return{value:null,done:!0}}},pE=class{constructor(){this.size=null}batch(e,t=!0){let n=this;b(e>0,()=>`batchSize needs to be positive, but it is
      ${e}`);let r;return r=this.size===1/0||this.size==null?this.size:t?Math.ceil(this.size/e):Math.floor(this.size/e),mE(async()=>(await n.iterator()).columnMajorBatch(e,t,hE),r)}concatenate(e){let t=this,n;return n=this.size===1/0||e.size===1/0?1/0:this.size!=null&&e.size!=null?this.size+e.size:null,mE(async()=>(await t.iterator()).concatenate(await e.iterator()),n)}filter(e){let t=this,n;return n=this.size===1/0?1/0:null,mE(async()=>(await t.iterator()).filter(t=>F(()=>e(t))),n)}async forEachAsync(e){return(await this.iterator()).forEachAsync(e)}map(e){let t=this;return mE(async()=>(await t.iterator()).map(t=>F(()=>e(t))),this.size)}mapAsync(e){let t=this;return mE(async()=>(await t.iterator()).mapAsync(e),this.size)}prefetch(e){if(e==null)throw RangeError("`Dataset.prefetch()` requires bufferSize to be specified.");let t=this;return mE(async()=>(await t.iterator()).prefetch(e),this.size)}repeat(e){let t=this,n;return n=this.size!=null&&e>0?this.size*e:e===0?0:this.size!=null&&(e===void 0||e<0)?1/0:null,mE(async()=>YT(JT(async()=>({value:await t.iterator(),done:!1})).take(e)),n)}skip(e){let t=this,n;return n=this.size!=null&&e>=0&&this.size>=e?this.size-e:this.size!=null&&(this.size<e||e===void 0||e<0)?0:null,mE(async()=>(await t.iterator()).skip(e),n)}shuffle(e,t,n=!0){if(e==null||e<0)throw this.size==null?RangeError("`Dataset.shuffle()` requires bufferSize to be specified."):RangeError(`\`Dataset.shuffle()\` requires bufferSize to be specified.  If your data fits in main memory (for regular JS objects), and/or GPU memory (for \`tf.Tensor\`s), consider setting bufferSize to the dataset size (${this.size} elements)`);let r=this,i=yd.alea(t||li().toString());return mE(async()=>{let t=i.int32();return n&&(t+=i.int32()),(await r.iterator()).shuffle(e,t.toString())},this.size)}take(e){let t=this,n;return n=this.size!=null&&this.size>e?e:this.size!=null&&this.size<=e?this.size:null,mE(async()=>(await t.iterator()).take(e),n)}async toArray(){if(this.size===1/0)throw Error(`Can not convert infinite data stream to array.`);return(await this.iterator()).toArray()}async toArrayForTest(){if(this.size===1/0)throw Error(`Can not convert infinite data stream to array.`);return(await this.iterator()).toArrayForTest()}};pE.MAX_BUFFER_SIZE=1e4;function mE(e,t=null){return new class extends pE{constructor(){super(...arguments),this.size=t}async iterator(){return e()}}}function hE(e){if(e===null)return null;let t=e[0];return VT(t)?{value:gE(e),recurse:!1}:{value:null,recurse:!0}}function gE(e){if(e.length===0)throw Error(`Can't make a batch of zero elements.`);return e[0]instanceof Ni?Cf(e):ma(e)}function J(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{e!=null&&b(e.dtype!==`complex64`,()=>`${t} does not support complex64 tensors in the CPU backend.`)})}var _E=qf,vE=class e extends p{nextDataId(){return e.nextDataId++}constructor(){super(),this.blockSize=48,this.firstUse=!0,this.data=new f(this,_a())}write(e,t,n){this.firstUse&&(this.firstUse=!1,A().get(`IS_NODE`)&&jr(`
============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
============================`));let r={id:this.nextDataId()};return this.data.set(r,{values:e,dtype:n,refCount:1}),r}makeTensorInfo(e,t,n){let r;if(t===`string`&&n!=null&&n.length>0&&ue(n[0])){let i=n.map(e=>ui(e));r=this.write(i,e,t)}else r=this.write(n,e,t);return{dataId:r,shape:e,dtype:t}}refCount(e){return this.data.has(e)?this.data.get(e).refCount:0}incRef(e){let t=this.data.get(e);t.refCount++}decRef(e){if(this.data.has(e)){let t=this.data.get(e);t.refCount--}}move(e,t,n,r,i){this.data.set(e,{values:t,dtype:r,refCount:i})}numDataIds(){return this.data.numDataIds()}async read(e){return this.readSync(e)}readSync(e){let{dtype:t,complexTensorInfos:n}=this.data.get(e);return t===`complex64`?Bh(this.readSync(n.real.dataId),this.readSync(n.imag.dataId)):ve(this.data.get(e).values,t)}bufferSync(e){let t=this.readSync(e.dataId);if(e.dtype===`string`)try{let n=t.map(e=>di(e));return _o(e.shape,e.dtype,n)}catch{throw Error(`Failed to decode encoded string bytes into utf-8`)}return _o(e.shape,e.dtype,t)}makeOutput(e,t,n){return _a().makeTensorFromTensorInfo(this.makeTensorInfo(t,n,e),this)}disposeData(e,t=!1){if(this.data.has(e)){if(this.data.get(e).refCount--,!t&&this.data.get(e).refCount>0)return!1;let{complexTensorInfos:n}=this.data.get(e);n!=null&&(this.disposeData(n.real.dataId,!0),this.disposeData(n.imag.dataId,!0)),this.data.delete(e)}return!0}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}async time(e){let t=li();return e(),{kernelMs:li()-t}}memory(){return{unreliable:!0,reasons:[`The reported memory is an upper bound. Due to automatic garbage collection, the true allocated memory may be less.`]}}where(e){J([e],`where`);let t=this.readSync(e.dataId);return _E(e.shape,t)}dispose(){}floatPrecision(){return 32}epsilon(){return super.epsilon()}};vE.nextDataId=0;function yE(e){let t=new Float32Array(e.length);for(let n=0;n<e.length;++n)t[n]=Math.abs(e[n]);return t}var bE={kernelName:`Abs`,backendName:`cpu`,kernelFunc:e=>{let{x:t}=e.inputs,n=e.backend;J(t,`abs`);let r=new Float32Array(C(t.shape)),i=n.data.get(t.dataId).values;return r=yE(i),n.makeOutput(r,t.shape,t.dtype)}};function xE(e){return(t,n,r,i,a)=>{let o=V(t,n),s=o.length,c=k(o),l=re(a,C(o)),u=t.length,d=n.length,f=k(t),p=k(n),m=Lc(t,o),h=Lc(n,o);if(m.length+h.length===0)for(let t=0;t<l.length;++t)l[t]=e(r[t%r.length],i[t%i.length]);else for(let t=0;t<l.length;++t){let n=we(t,s,c),a=n.slice(-u);m.forEach(e=>a[e]=0);let o=Ce(a,u,f),g=n.slice(-d);h.forEach(e=>g[e]=0);let _=Ce(g,d,p);l[t]=e(r[o],i[_])}return[l,o]}}function SE(e){let{inputs:t,backend:n}=e,{real:r,imag:i}=t,a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,s=n.makeTensorInfo(r.shape,`complex64`),c=n.data.get(s.dataId);return c.complexTensorInfos={real:n.makeTensorInfo(r.shape,`float32`,a),imag:n.makeTensorInfo(i.shape,`float32`,o)},s}var CE={kernelName:ot,backendName:`cpu`,kernelFunc:SE};function wE(e,t,n=`float32`){if(n===`complex64`)return SE({inputs:{real:wE(e,t,`float32`),imag:wE(e,t,`float32`)},backend:e});let r=be(C(t),n);return e.makeTensorInfo(t,n,r)}function TE(e){let{inputs:t,backend:n}=e,{x:r}=t;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}var EE={kernelName:Wt,backendName:`cpu`,kernelFunc:TE};function DE(e){let{inputs:t,backend:n}=e,{input:r}=t,i=n.data.get(r.dataId).complexTensorInfos.real,a=n.data.get(i.dataId).values;return n.makeTensorInfo(i.shape,i.dtype,a)}var OE={kernelName:Mn,backendName:`cpu`,kernelFunc:DE};function kE(e,t,n,r){if(r===`int32`)return[t,`int32`,Int32Array.from(e)];if(r===`bool`){let r=ci([0],n),[i,a]=xE((e,t)=>e===t?0:1)(t,[],e,r,`bool`);return[a,`bool`,i]}throw Error(`Error in Cast: failed to cast ${n} to ${r}`)}function AE(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dtype:a}=r;if(a===`complex64`){if(i.dtype===`complex64`)return TE({inputs:{x:i},backend:n});let e=wE(n,i.shape,i.dtype),t=AE({inputs:{x:i},backend:n,attrs:{dtype:`float32`}}),r=SE({inputs:{real:t,imag:e},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),r}if(i.dtype===`complex64`){let e=DE({inputs:{input:i},backend:n}),t=AE({inputs:{x:e},backend:n,attrs:{dtype:a}});return n.disposeIntermediateTensorInfo(e),t}if(!se(i.dtype,a)){let e=TE({inputs:{x:i},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:a}}let o=n.data.get(i.dataId).values,[s,c,l]=kE(o,i.shape,i.dtype,a);return n.makeTensorInfo(s,c,l)}var jE={kernelName:rt,backendName:`cpu`,kernelFunc:AE};function ME(e,t,n,r){return n==null?({inputs:n,backend:i})=>{let{a,b:o}=n,s=i;J([a,o],e);let c=s.data.get(a.dataId).values,l=s.data.get(o.dataId).values,u=a.dtype===`string`?Sg(c):c,d=a.dtype===`string`?Sg(l):l,f=r||a.dtype,[p,m]=t(a.shape,o.shape,u,d,f);return s.makeTensorInfo(m,f,p)}:({inputs:e,backend:i})=>{let{a,b:o}=e,s=i;if(a.dtype===`complex64`||o.dtype===`complex64`){let e=AE({inputs:{x:a},backend:s,attrs:{dtype:`complex64`}}),t=s.data.get(e.dataId),r=t.complexTensorInfos.real,i=t.complexTensorInfos.imag,c=s.data.get(r.dataId).values,l=s.data.get(i.dataId).values,u=AE({inputs:{x:o},backend:s,attrs:{dtype:`complex64`}}),d=s.data.get(u.dataId),f=d.complexTensorInfos.real,p=d.complexTensorInfos.imag,m=s.data.get(f.dataId).values,h=s.data.get(p.dataId).values,[g,_,v]=n(a.shape,o.shape,c,l,m,h),y=s.makeTensorInfo(v,`float32`,g),b=s.makeTensorInfo(v,`float32`,_),x=SE({inputs:{real:y,imag:b},backend:s});return s.disposeIntermediateTensorInfo(e),s.disposeIntermediateTensorInfo(u),s.disposeIntermediateTensorInfo(y),s.disposeIntermediateTensorInfo(b),x}{let e=s.data.get(a.dataId).values,n=s.data.get(o.dataId).values,i=r||a.dtype,[c,l]=t(a.shape,o.shape,e,n,i);return s.makeTensorInfo(l,i,c)}}}function NE(e){return(t,n,r,i,a,o)=>{let s=V(t,n),c=C(s),l=s.length,u=k(s),d=re(`float32`,c),f=re(`float32`,c),p=Lc(t,s),m=Lc(n,s),h=Bh(r,i),g=Bh(a,o),_=t.length,v=k(t),y=n.length,b=k(n);if(p.length+m.length===0)for(let t=0;t<d.length;t++){let n=t%h.length,r=t%g.length,i=e(h[n*2],h[n*2+1],g[r*2],g[r*2+1]);d[t]=i.real,f[t]=i.imag}else for(let t=0;t<d.length;t++){let n=we(t,l,u),r=n.slice(-_);p.forEach(e=>r[e]=0);let i=Ce(r,_,v),a=n.slice(-y);m.forEach(e=>a[e]=0);let o=Ce(a,y,b),s=e(h[i*2],h[i*2+1],g[o*2],g[o*2+1]);d[t]=s.real,f[t]=s.imag}return[d,f,s]}}var PE=xE(((e,t)=>e+t)),FE=ME(`Add`,PE,NE(((e,t,n,r)=>({real:e+n,imag:t+r})))),IE={kernelName:`Add`,backendName:`cpu`,kernelFunc:FE};function LE(e,t,n,r,i){let a=C(r),o=be(i,n);for(let n=0;n<e.length;n++){let r=e[n];if(r<0)throw Error(`Input x must be non-negative!`);r>=i||(a>0?o[r]+=t[n]:o[r]+=1)}return o}function RE(e,t,n,r=!1){let i=e.shape[0],a=e.shape[1],o=_o([i,n],t.dtype);for(let s=0;s<i;s++)for(let i=0;i<a;i++){let a=e.get(s,i);if(a<0)throw Error(`Input x must be non-negative!`);a>=n||(r?o.set(1,s,a):t.size>0?o.set(o.get(s,a)+t.get(s,i),s,a):o.set(o.get(s,a)+1,s,a))}return o}var zE=xE(((e,t)=>e&t)),BE={kernelName:et,backendName:`cpu`,kernelFunc:ME(et,zE)};function VE(e){return(t,n,r)=>{let i=ie(n,t.length);for(let n=0;n<t.length;++n)i[n]=e(t[n],r);return i}}function HE(e,t,n){return UE(e,VE(t),n)}function UE(e,t,n){return({inputs:r,attrs:i,backend:a})=>{let{x:o}=r;J(o,e);let s=a,c=s.data.get(o.dataId).values,l;if(o.dtype===`string`){if(!Array.isArray(c))throw Error(`String tensor's value was not an instance of Array`);l=Sg(c)}else l=c;let u=n||o.dtype,d=t(l,u,i);return s.makeTensorInfo(o.shape,u,d)}}var WE=VE(e=>Math.ceil(e)),GE={kernelName:it,backendName:`cpu`,kernelFunc:UE(it,WE)};function KE(e,t,n,r){let i=ie(n,C(t));if(r&&n!==`string`){let t=0;e.forEach(e=>{let n=C(e.shape);i.set(e.vals,t),t+=n})}else{let r=0;e.forEach(e=>{let a=n===`string`?Sg(e.vals):e.vals,o=0;for(let n=0;n<e.shape[0];++n){let s=n*t[1]+r;for(let t=0;t<e.shape[1];++t)i[s+t]=a[o++]}r+=e.shape[1]})}return i}var qE=xE((e,t)=>+(e===t)),JE=ME(Mt,qE,null,`bool`),YE={kernelName:Mt,backendName:`cpu`,kernelFunc:JE},XE=VE(e=>Math.exp(e)),ZE=UE(`Exp`,XE,`float32`),QE={kernelName:`Exp`,backendName:`cpu`,kernelFunc:ZE},$E=VE(e=>Math.expm1(e)),eD={kernelName:Pt,backendName:`cpu`,kernelFunc:UE(Pt,$E)},tD=VE(e=>Math.floor(e)),nD={kernelName:Lt,backendName:`cpu`,kernelFunc:UE(Lt,tD)},rD=xE((e,t)=>Math.floor(e/t)),iD={kernelName:Rt,backendName:`cpu`,kernelFunc:ME(Rt,rD,null,`int32`)};function aD(e,t,n,r,i,a,o,s,c){let l=_o([r,a],n);for(let n=0;n<r;n++){let r=[],u=0;for(let t=0;t<i;t++){let a=e[n*i+t];u+=a*o[t],r.push(a)}if(u<0||u>=c/a)throw Error(`Invalid indices: ${r} does not index into ${s}`);for(let e=0;e<a;e++)l.values[n*a+e]=t.get(...t.indexToLoc(u*a+e))}return l}function oD(e,t,n){let r=_o(n,e.dtype);for(let n=0;n<r.size;++n){let i=r.indexToLoc(n).slice(),a=i[0],o=i[2],s=t.locToIndex([a,o]);i[2]=t.values[s];let c=e.locToIndex(i);0<=c&&c<e.values.length&&(r.values[n]=e.values[c])}return r}var sD=xE((e,t)=>+(e>t)),cD={kernelName:Ht,backendName:`cpu`,kernelFunc:ME(Ht,sD,null,`bool`)},lD=xE((e,t)=>+(e>=t)),uD={kernelName:Ut,backendName:`cpu`,kernelFunc:ME(Ut,lD,null,`bool`)},dD=xE((e,t)=>+(e<t)),fD={kernelName:Zt,backendName:`cpu`,kernelFunc:ME(Zt,dD,null,`bool`)},pD=xE((e,t)=>+(e<=t)),mD={kernelName:Qt,backendName:`cpu`,kernelFunc:ME(Qt,pD,null,`bool`)};function hD(e,t,n){let r=(t-e)/(n-1),i=be(n,`float32`);i[0]=e;for(let e=1;e<i.length;e++)i[e]=i[e-1]+r;return i}var gD=VE(e=>Math.log(e)),_D={kernelName:`Log`,backendName:`cpu`,kernelFunc:UE(`Log`,gD)};function vD(e,t,n,r){let i=re(r,C(n));for(let n=0;n<i.length;++n){let r=n*t,a=e[r];for(let n=0;n<t;++n){let t=e[r+n];(Number.isNaN(t)||t>a)&&(a=t)}i[n]=a}return i}var yD=xE(((e,t)=>Math.max(e,t))),bD={kernelName:sn,backendName:`cpu`,kernelFunc:ME(sn,yD)},xD=xE(((e,t)=>Math.min(e,t))),SD={kernelName:mn,backendName:`cpu`,kernelFunc:ME(mn,xD)},CD=xE(((e,t)=>e*t)),wD=ME(_n,CD,NE(((e,t,n,r)=>({real:e*n-t*r,imag:e*r+t*n})))),TD={kernelName:_n,backendName:`cpu`,kernelFunc:wD};function ED(e,t,n){return CD([],t,oi(-1,n),e,n)}function DD(e){let{inputs:t,backend:n}=e,{x:r}=t;J(r,`neg`);let i=n.data.get(r.dataId).values,[a,o]=ED(i,r.shape,r.dtype);return n.makeTensorInfo(o,r.dtype,a)}var OD={kernelName:`Neg`,backendName:`cpu`,kernelFunc:DD},kD=xE(((e,t)=>e===t?0:1)),AD={kernelName:vn,backendName:`cpu`,kernelFunc:ME(vn,kD,null,`bool`)};function jD(e,t,n,r,i){let a=t.length,o=C(t),s=k(t),c=k(i),l=re(n,C(i));for(let t=0;t<o;++t){let n=we(t,a,s),i=Array(n.length);for(let e=0;e<i.length;e++)i[e]=n[r[e]];let o=Ce(i,a,c);l[o]=e[t]}return l}function MD(e){let{inputs:t,attrs:n,backend:r}=e,{x:i}=t,{perm:a}=n;J(i,`transpose`);let o=i.shape.length,s=Array(o);for(let e=0;e<s.length;e++)s[e]=i.shape[a[e]];let c=r.data.get(i.dataId).values,l=jD(c,i.shape,i.dtype,a,s);return{dataId:r.write(l,s,i.dtype),shape:s,dtype:i.dtype}}var ND={kernelName:br,backendName:`cpu`,kernelFunc:MD};function PD(e,t,n,r){let[i,a]=rl(e,r),o=Vi(t,`int32`),s=be(C(i),o),c=C(a);for(let e=0;e<s.length;++e){let t=e*c,r=1;for(let e=0;e<c;++e)r*=n[t+e];s[e]=r}return{outVals:s,outShape:i,outDtype:o}}function FD(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;J(i,`prod`);let s=i.shape.length,c=O(a,i.shape),l=ol(c,s),u=c,d=i,f=[];l!=null&&(d=MD({inputs:{x:i},backend:n,attrs:{perm:l}}),f.push(d),u=cl(u.length,s));let p=n.data.get(d.dataId).values,{outVals:m,outShape:h,outDtype:g}=PD(d.shape,d.dtype,p,u),_=h;return o&&(_=il(h,c)),f.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(_,g,m)}var ID={kernelName:Dn,backendName:`cpu`,kernelFunc:FD};function LD(e,t,n){e.forEach((e,r)=>{if(e<0||e>=n){let i=we(r,t.length,k(t)).join(`,`);throw Error(`indices[${i}] = ${e} is not in [0, ${n})`)}})}function RD(e,t){for(let n=0;n<e.length;++n){let r=e[n],i=n===e.length-1?t:e[n+1].length;if(r.length===0)throw Error(`Ragged splits may not be empty`);if(r[0]<0)throw Error(`Ragged splits must be non-negative`);if(r[r.length-1]>i)throw Error(`Ragged splits must not point past values`);for(let e=1;e<r.length;++e)if(r[e-1]>r[e])throw Error(`Ragged splits must be sorted in ascending order`)}}function zD(e,t,n,r){let i=[],a=0,o=t.length-1+n.length,s=Array(o).fill(null).map(()=>[0]);RD(n,r);let c=1;for(let e=0;e<t.length-1;++e){c*=t[e];let n=t[e+1];for(let t=1;t<c+1;++t)s[e].push(t*n)}for(let r=0;r<e.length;++r){let o=e[r],c=e[r]+1;for(let e=0;e<n.length;++e){let r=n[e],i=e+t.length-1;if(i>=0){let e=s[i],t=e[e.length-1]-r[o];for(let e=o;e<c;++e)s[i].push(r[e+1]+t)}o=r[o],c=r[c]}c!==o&&(i.push([o,c]),a+=c-o)}return{outSplits:s,valueSlices:i,numValues:a}}function BD(e){let t=[];for(let n=0;n<e.length;++n){let r=e[n].length,i=ie(`int32`,r);t.push(i),e[n].forEach((e,t)=>i[t]=e)}return t}function VD(e,t){let n=e.slice(0,t);for(;n.length<t;)n.push(1);for(let r=t;r<e.length;r++)n[t-1]*=e[r];return n}function HD(e,t,n,r,i,a){let o=VD(t,2)[1],s=VD(a,2)[1],c=0;for(let t of n)for(let n=t[0];n<t[1];++n){for(let t=0;t<r;++t)i[c*s+t]=e[n*o+t];++c}}function UD(e,t,n,r,i){let a=t.slice();a[0]=i;let o=ie(n,C(a)),s=e.length;return HD(e,t,r,s===0?0:s/t[0],o,a),[o,a]}function WD(e,t,n,r,i,a,o,s){if(e.length===0)throw Error(`paramsNestedSplits must be non empty`);if(t[0].length===0)throw Error(`Split tensors must not be scalars`);if(LD(a,o,t[0][0]-1),r.length===0)throw Error(`params.rank must be nonzero`);let c=r[0],{outSplits:l,valueSlices:u,numValues:d}=zD(a,o,e,c),f=BD(l),p=UD(n,r,i,u,d);return[f,p[0],p[1]]}var GD=2147483647;function KD(e,t,n,r,i,a,o){if(t.length>1)throw Error(`starts must be a scalar or vector`);if(i.length>1)throw Error(`limits must be a scalar or vector`);if(o.length>1)throw Error(`deltas must be a scalar or vector`);let s=t.length===0,c=i.length===0,l=o.length===0,u=[];s||u.push(t[0]),c||u.push(i[0]),l||u.push(o[0]);for(let e=1;e<u.length;++e)if(u[e]!==u[e-1])throw Error(`starts, limits, and deltas must have the same shape`);let d=u.length===0?1:u[0],f=ie(`int32`,d+1);f[0]=0;for(let t=0;t<d;++t){let n=s?e[0]:e[t],i=c?r[0]:r[t],o=l?a[0]:a[t];if(o===0)throw Error(`Requires delta != 0`);let u;if(o>0&&i<n||o<0&&i>n)u=0;else if(u=Math.ceil(Math.abs((i-n)/o)),u>GD)throw Error(`Requires ((limit - start) / delta) <= ${GD}`);f[t+1]=f[t]+u}let p=f[d],m=ie(n,p),h=0;for(let t=0;t<d;++t){let n=f[t+1]-f[t],r=s?e[0]:e[t],i=l?a[0]:a[t];for(let e=0;e<n;++e)m[h++]=r,r+=i}return[f,m]}var qD=bh,JD=class e{constructor(e,t,n,r,i,a,o,s,c,l){this.shape=e,this.shapeShape=t,this.values=n,this.valuesShape=r,this.valuesDType=i,this.defaultValue=a,this.defaultValueShape=o,this.rowPartitionValues=s,this.rowPartitionValuesShapes=c,this.rowPartitionTypes=Sh(l),this.raggedRank=Ch(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===qD.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===qD.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(t){let n=this.getRowPartitionTensor(t-1);switch(this.getRowPartitionTypeByDimension(t-1)){case qD.VALUE_ROWIDS:return e.getMaxWidthValueRowID(n);case qD.ROW_SPLITS:return e.getMaxWidthRowSplit(n);default:throw Error(`Cannot handle partition type ${qD[this.getRowPartitionTypeByDimension(t-1)]}`)}}static getMaxWidthRowSplit(e){let t=e.length;if(t===0||t===1)return 0;let n=0;for(let r=0;r<t-1;++r){let t=e[r+1]-e[r];t>n&&(n=t)}return n}static getMaxWidthValueRowID(e){let t=e.length;if(t===0)return 0;let n=0,r=e[0],i=0;for(let a=1;a<t;++a){let t=e[a];t!==r&&(r=t,i=Math.max(a-n,i),n=a)}return Math.max(t-n,i)}tensorShapeFromTensor(e,t,n=!0){if(t.length===0){if(e[0]===-1)return[];throw Error(`The only valid scalar shape tensor is the fully unknown shape specified as -1.`)}return XD(e,n)}calculateOutputSize(e){let t=this.valuesShape,n=this.defaultValueShape;wh(n,t);let r=this.tensorShapeFromTensor(this.shape,this.shapeShape),i=xh(this.raggedRank,r,t);i[0]<0&&(i[0]=e);for(let e=1;e<=this.raggedRank;++e)i[e]<0&&(i[e]=this.getMaxWidth(e));return i}calculateFirstParentOutputIndex(e,t,n){let r=Math.min(e,n),i=[],a=0;for(let e=0;e<r;++e,a+=t)i.push(a);for(let t=r;t<e;++t)i.push(-1);return b(i.length===e,()=>`Final length of result must be equal to firstDimension.`),i}calculateOutputIndexRowSplit(e,t,n,r){let i=e.length,a=[];for(let o=0;o<i-1;++o){let i=e[o+1]-e[o],s=Math.min(r,i),c=t[o];c===-1&&(s=0);for(let e=0;e<s;++e)a.push(c),c+=n;for(let e=0;e<i-s;++e)a.push(-1)}if(i>0&&a.length!==e[i-1])throw Error(`Invalid row split size.`);return a}calculateOutputIndexValueRowID(e,t,n,r){let i=e.length,a=[];if(i===0)return[];let o=0,s=e[0];if(s>=t.length)throw Error(`Got currentValueRowId=${s}, which is not less than ${t.length}`);let c=t[s];a.push(c);for(let l=1;l<i;++l){let i=e[l];if(i===s)c>=0&&(++o,o<r?c+=n:c=-1);else{if(o=0,s=i,i>=t.length)throw Error(`Got nextValueRowId=${i} which is not less than ${t.length}`);c=t[i]}a.push(c)}if(a.length!==e.length)throw Error(`Invalid row ids.`);return a}calculateOutputIndex(e,t,n,r){let i=this.getRowPartitionTensor(e),a=this.getRowPartitionTypeByDimension(e);switch(a){case qD.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(i,t,n,r);case qD.ROW_SPLITS:if(i.length-1>t.length)throw Error(`Row partition size is greater than output size: ${i.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(i,t,n,r);default:throw Error(`Unsupported partition type: ${qD[a]}`)}}getFirstDimensionSize(){let e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw Error(`No row_partition_types given.`);let t=this.rowPartitionTypes[0];switch(t){case qD.FIRST_DIM_SIZE:return e[0];case qD.VALUE_ROWIDS:throw Error(`Cannot handle VALUE_ROWIDS in first dimension.`);case qD.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw Error(`Cannot handle type ${qD[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw Error(`Invalid first partition input. Tensor requires at least one element.`);let e=this.getFirstDimensionSize(),t=this.calculateOutputSize(e),n=Array(this.raggedRank+1);n[n.length-1]=1;for(let e=n.length-2;e>=0;--e)n[e]=n[e+1]*t[e+1];let r=XD(t,!1),i=ie(this.valuesDType,C(r));if(n[0]*t[0]>0){let a=this.calculateFirstParentOutputIndex(e,n[0],t[0]);for(let e=1;e<=this.raggedRank;++e)a=this.calculateOutputIndex(e-1,a,n[e],t[e]);this.setOutput(this.raggedRank,a,i,r)}return[r,i]}setOutput(e,t,n,r){if(n.length===0)return;let i=this.values,a=n,o=r.slice();o=o.slice(e+1);let s=C(o),c=t.length,l=this.defaultValue;if(l.length!==s&&l.length!==1){let e=this.defaultValueShape;F(()=>{l=qs(B(l,e),o).dataSync()})}let u=0,d=0,f=0;for(let e=0;e<=c;++e){let r=e<c?t[e]:-1;if(r===f){++f;continue}if(d<f){let e=i.subarray(u*s);YD(a.subarray(d*s),e,(f-d)*s)}if(e>=c){let e=n.length;r=Math.floor(e/s)}if(r>f){if(this.defaultValue.length===1)a.subarray(f*s,r*s).fill(this.defaultValue[0]),f=r;else for(;r>f;)YD(a.slice(f*s),l,s),++f}r<0?(u=e+1,d=f):(u=e,d=f,f=d+1)}}};function YD(e,t,n){for(let r=0;r<n;r++)e[r]=t[r]}function XD(e,t){let n=[];for(let r of e){if(r<0){if(!t)throw Error(`Dimension ${r} must be >= 0`);if(r<-1)throw Error(`Dimension ${r} must be >= -1`);r=-1}n.push(r)}return n}function ZD(e,t,n,r,i,a,o,s,c,l){return new JD(e,t,n,r,i,a,o,s,c,l).compute()}function QD(e,t,n,r){if(e===t||e<t&&n<0||t<e&&n>1)return be(0,r);let i=be(Math.abs(Math.ceil((t-e)/n)),r);t<e&&n===1&&(n=-1),i[0]=e;for(let e=1;e<i.length;e++)i[e]=i[e-1]+n;return i}var $D=VE(e=>1/Math.sqrt(e)),eO={kernelName:Un,backendName:`cpu`,kernelFunc:UE(Un,$D)};function tO(e,t,n,r,i,a,o,s,c,l){let u=[r/i,i],d=e.values,f=t.values;if(r===0)return _o(n,t.dtype);let p=c instanceof Oi?c:_o(u,t.dtype);typeof c==`string`||typeof c==`number`?p.values.fill(c):typeof c==`boolean`&&p.values.fill(+c);for(let e=0;e<a;e++){let a=[],c=0;for(let t=0;t<o;t++){let n=d[e*o+t];a.push(n),c+=n*s[t]}if(c<0||c>=r/i)throw Error(`Invalid indices: ${a} does not index into ${n}`);for(let n=0;n<i;n++)l?p.values[c*i+n]+=f[e*i+n]:p.values[c*i+n]=t.rank===0?f[0]:f[e*i+n]}return p}var nO=VE(e=>1/(1+Math.exp(-e))),rO=HE(Qn,e=>1/(1+Math.exp(-e))),iO={kernelName:Qn,backendName:`cpu`,kernelFunc:rO};function aO(e,t,n,r,i){let a=lh(r,t,n),o=C(n),s=k(r);if(a){let n=uh(t,s);return i===`string`?e.slice(n,n+o):e.subarray(n,n+o)}let c=_o(r,i,i===`string`?Sg(e):e),l=_o(n,i);for(let e=0;e<l.size;++e){let n=l.indexToLoc(e),r=n.map((e,n)=>e+t[n]);l.set(c.get(...r),...n)}return i===`string`?Cg(l.values):l.values}function oO(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,size:o}=r;J(i,`slice`);let[s,c]=dh(i,a,o);Zm(i,s,c);let l=n.data.get(i.dataId).values,u=aO(l,s,c,i.shape,i.dtype);return n.makeTensorInfo(c,i.dtype,u)}var sO={kernelName:Yn,backendName:`cpu`,kernelFunc:oO};function cO(e,t,n,r,i,a,o){let s=t[0],c=a[0],l=Array(c),u=Array(s),d=t[1];if(c===0){if(s!==0)throw Error(ag(s));let e=ie(n,0),t=ie(i,0);return[e,[0,d],t,l,u]}let f=!0,p=0,m=Array(c).fill(0);for(let t=0;t<s;++t){let n=e[t*d];if(n<0)throw Error(og(t,n));if(n>=c)throw Error(sg(t,n,c));++m[n],f&&=n>=p,p=n}let h=!0;for(let e=0;e<c;++e){let t=m[e]===0;l[e]=t,h&&=!t,m[e]=Math.max(m[e],1),e>0&&(m[e]+=m[e-1])}if(h&&f){let t=e,n=r;for(let e=0;e<s;++e)u[e]=e;return[t,[s,d],n,l,u]}{let t=m[c-1],a=ie(n,t*d),f=ie(i,t),p=Array(c).fill(0);for(let t=0;t<s;++t){let n=e[t*d],i=p[n],o=(n===0?0:m[n-1])+i;p[n]++;for(let n=0;n<d;++n)a[o*d+n]=e[t*d+n];f[o]=r[t],u[t]=o}for(let e=0;e<c;++e)if(p[e]===0){let t=e===0?0:m[e-1];a[t*d+0]=e;for(let e=1;e<d;++e)a[t*d+e]=0;f[t]=o}return[a,[t,d],f,l,u]}}function lO(e,t,n,r,i){let a=C(r),o=t[0],s=i.length,c=[],l=1,u=-1;for(let e=0;e<s;++e){let t=i[e];if(t===-1){if(u!==-1)throw Error(cg(u,e));u=e,c.push(1)}else{if(t<0)throw Error(lg(e,t));l*=t,c.push(t)}}if(u!==-1){if(l<=0)throw Error(ug());let e=Math.trunc(a/l);if(l*e!==a)throw Error(dg(r,c));c[u]=e}if(C(c)!==a)throw Error(fg(r,c));let d=r.length,f=[];if(d>0){f[d-1]=1;for(let e=d-2;e>=0;--e)f[e]=f[e+1]*r[e+1]}let p=[];if(s>0){p[s-1]=1;for(let e=s-2;e>=0;--e)p[e]=p[e+1]*c[e+1]}let m=ie(n,o*s);for(let t=0;t<o;++t){let n=0;for(let r=0;r<d;++r)n+=e[t*d+r]*f[r];for(let e=0;e<s;++e)m[t*s+e]=Math.trunc(n/p[e]),n%=p[e]}return[m,[o,s],c]}function uO(e,t,n,r,i,a=!1,o=0){let s=r.length,c=[t[0],e.length/t[0]],l=c[1],u=s>0?i[s-1]+1:0;if(u<0)throw Error(pg());let d=t.slice();d[0]=u;let f=ie(n,d.reduce((e,t)=>e*t,1));if(s===0)return u>0&&f.fill(o),[f,d];if(u<=0)throw Error(pg());let p=0,m=1,h=0,g=i[p];for(;;){let t=0;if(m<s){if(t=i[m],g===t){++m;continue}if(g>=t)throw Error(mg())}if(g<0||g>=u)throw Error(hg(g,u));g>h&&f.fill(o,h*l,g*l);for(let t=p;t<m;++t){let n=r[t];if(n<0||n>=c[0])throw Error(gg(t,r[t],c[0]));for(let t=0;t<l;t++)f[g*l+t]+=e[n*l+t]}if(a)for(let e=0;e<l;e++)f[g*l+e]/=m-p;if(p=m,++m,h=g+1,g=t,m>s)break}return h<u&&f.fill(o,h*l,u*l),[f,d]}var dO=VE(e=>Math.sqrt(e)),fO={kernelName:er,backendName:`cpu`,kernelFunc:HE(er,e=>Math.sqrt(e))},pO=xE(((e,t)=>{let n=e-t;return n*n})),mO={kernelName:lr,backendName:`cpu`,kernelFunc:ME(lr,pO)},hO=VE((e,t)=>{let{pattern:n,replaceGlobal:r,rewrite:i}=t;return e.replace(new RegExp(n,r?`g`:``),i)}),gO={kernelName:dr,backendName:`cpu`,kernelFunc:UE(dr,hO)};function _O(e,t,n,r){let i=_o(e,t.dtype);for(let e=0;e<i.size;e++){let a=i.indexToLoc(e),o=Array(a.length);for(let e=0;e<o.length;e++)o[e]=a[e]*n[e]+r[e];i.set(t.get(...o),...a)}return i}var vO=class{constructor(e,t,n,r,i,a){this.separator=ui(e),this.nGramWidths=t,this.leftPad=ui(n),this.rightPad=ui(r),this.padWidth=i,this.preserveShort=a}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){let n=this.getPadWidth(t);return Math.max(0,e+2*n-t+1)}createNGrams(e,t,n,r,i,a){for(let o=0;o<i;++o){let s=this.getPadWidth(a),c=Math.max(0,s-o),l=Math.max(0,s-(i-(o+1))),u=a-(c+l),d=t+(c>0?0:o-s),f=0;f+=c*this.leftPad.length;for(let t=0;t<u;++t)f+=e[d+t].length;f+=l*this.rightPad.length;let p=c+l+u-1;f+=p*this.separator.length,n[r+o]=new Uint8Array(f);let m=n[r+o],h=0,g=e=>e.forEach(e=>m[h++]=e);for(let e=0;e<c;++e)g(this.leftPad),g(this.separator);for(let t=0;t<u-1;++t)g(e[d+t]),g(this.separator);if(u>0){g(e[d+u-1]);for(let e=0;e<l;++e)g(this.separator),g(this.rightPad)}else{for(let e=0;e<l-1;++e)g(this.rightPad),g(this.separator);g(this.rightPad)}}}compute(e,t){let n=e.length,r=t.length;if(r>0){let e=t[0];if(e!==0)throw Error(`First split value must be 0, got ${e}`);for(let i=1;i<r;++i){let r=t[i]>=e;if(r&&=t[i]<=n,!r)throw Error(`Invalid split value ${t[i]}, must be in [${e}, ${n}]`);e=t[i]}if(e!==n)throw Error(`Last split value must be data size. Expected ${n}, got ${e}`)}let i=r-1,a=ie(`int32`,r);if(n===0||r===0){let e=Array(n);for(let e=0;e<=i;++e)a[e]=0;return[e,a]}a[0]=0;for(let e=1;e<=i;++e){let n=t[e]-t[e-1],r=0;this.nGramWidths.forEach(e=>{r+=this.getNumNGrams(n,e)}),this.preserveShort&&n>0&&r===0&&(r=1),a[e]=a[e-1]+r}let o=Array(a[i]);for(let n=0;n<i;++n){let r=t[n],i=a[n];if(this.nGramWidths.forEach(a=>{let s=t[n+1]-t[n],c=this.getNumNGrams(s,a);this.createNGrams(e,r,o,i,c,a),i+=c}),this.preserveShort&&i===a[n]){let a=t[n+1]-t[n];if(a===0)continue;let s=a+2*this.padWidth;this.createNGrams(e,r,o,i,1,s)}}return[o,a]}};function yO(e,t,n,r,i,a,o,s){return new vO(n,r,i,a,o,s).compute(e,t)}function bO(e,t,n,r){if(!e.length)return;if(t.length===0){for(let t=0;t<e.length;++t)r.push(e.subarray(t,t+1));return}if(t.length===1){let i=t[0],a=e.indexOf(i);for(;a!==-1;){let t=e.subarray(0,a);(!n||t.length!==0)&&r.push(t),e=e.subarray(a+1),a=e.indexOf(i)}(!n||e.length!==0)&&r.push(e);return}let i=0;for(let a=0;a<e.length+1;a++)if(a===e.length||t.indexOf(e[a])!==-1){let t=e.subarray(i,a);(!n||t.length!==0)&&r.push(t),i=a+1}}function xO(e,t,n){let r=e.length,i=[],a=0,o=0,s=Array(r);for(let c=0;c<r;++c){let r=i.length;bO(e[c],t,n,i);let l=i.length-r;s[c]=l,a+=l,o=Math.max(o,l)}let c=ie(`int32`,a*2),l=Array(a),u=[r,o],d=0;for(let e=0;e<r;++e)for(let t=0;t<s[e];++t)c[d*2]=e,c[d*2+1]=t,l[d]=i[d],++d;return[c,l,u]}function SO(e,t){let n=ie(`int32`,e.length);for(let r=0;r<e.length;++r)n[r]=ai(e[r]).modulo(t).getLowBitsUnsigned();return n}var CO=xE(((e,t)=>e-t)),wO=ME(`Sub`,CO,NE(((e,t,n,r)=>({real:e-n,imag:t-r})))),TO={kernelName:`Sub`,backendName:`cpu`,kernelFunc:wO};function EO(e,t){let n=Array(e.rank);for(let r=0;r<n.length;r++)n[r]=e.shape[r]*t[r];let r=_o(n,e.dtype);for(let t=0;t<r.values.length;++t){let n=r.indexToLoc(t),i=Array(e.rank);for(let t=0;t<i.length;t++)i[t]=n[t]%e.shape[t];let a=e.locToIndex(i);r.values[t]=e.values[a]}return r}var DO=(e,t)=>{let n=t.value-e.value;return n===0?e.index-t.index:n};function OO(e,t,n=0,r=e.length-1){for(;r>n;){if(r-n>600){let i=r-n+1,a=t-n+1,o=Math.log(i),s=.5*Math.exp(2*o/3),c=.5*Math.sqrt(o*s*(i-s)/i)*Math.sign(a-i/2);OO(e,t,Math.max(n,Math.floor(t-a*s/i+c)),Math.min(r,Math.floor(t+(i-a)*s/i+c)))}let i=e[t],a=n,o=r;for(v(e,n,t),DO(e[r],i)>0&&v(e,n,r);a<o;){for(v(e,a,o),a++,o--;DO(e[a],i)<0;)a+=1;for(;DO(e[o],i)>0;)--o}DO(e[n],i)===0?v(e,n,o):(o+=1,v(e,o,r)),o<=t&&(n=o+1),t<=o&&(r=o-1)}}function kO(e,t,n,r,i){let a=t[t.length-1],[o,s]=[e.length/a,a],c=re(n,o*r),l=re(`int32`,o*r);for(let t=0;t<o;t++){let n=t*s,a=e.subarray(n,n+s),o=Array(a.length);a.forEach((e,t)=>o[t]={value:e,index:t}),r<o.length&&(OO(o,r),o=o.slice(0,r)),i&&o.sort(DO);let u=t*r,d=c.subarray(u,u+r),f=l.subarray(u,u+r);for(let e=0;e<r;e++)d[e]=o[e].value,f[e]=o[e].index}let u=t.slice();return u[u.length-1]=r,[_o(u,n,c),_o(u,`int32`,l)]}function AO(e,t,n,r){let i=O(t,n)[0],a=[1,n[0],1];for(let e=0;e<i;e++)a[0]*=n[e];a[1]=n[i];for(let e=i+1;e<n.length;e++)a[2]*=n[e];let o=new Map,s=new Int32Array(n[i]),c=new Oi(a,r,e),l=[],u=a[0]===1&&a[2]===1;for(let t=0;t<n[i];t++){let n;if(u)n=e[t].toString();else{let e=[];for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)e.push(c.get(n,t,r));n=e.join(`,`)}let r=o.get(n);if(r!=null)s[t]=r;else{let e=o.size;o.set(n,e),s[t]=e,l.push(t)}}let d=a.slice();d[1]=o.size;let f=new Oi(d,r);l.forEach((e,t)=>{for(let n=0;n<a[0];n++)for(let r=0;r<a[2];r++)f.set(c.get(n,e,r),n,t,r)});let p=n.slice();return p[i]=d[1],{outputValues:f.values,outputShape:p,indices:s}}var jO=s({addImpl:()=>PE,bincountImpl:()=>LE,bincountReduceImpl:()=>RE,bitwiseAndImpl:()=>zE,castImpl:()=>kE,ceilImpl:()=>WE,concatImpl:()=>KE,equalImpl:()=>qE,expImpl:()=>XE,expm1Impl:()=>$E,floorDivImpl:()=>rD,floorImpl:()=>tD,gatherNdImpl:()=>aD,gatherV2Impl:()=>oD,greaterEqualImpl:()=>lD,greaterImpl:()=>sD,lessEqualImpl:()=>pD,lessImpl:()=>dD,linSpaceImpl:()=>hD,logImpl:()=>gD,maxImpl:()=>vD,maximumImpl:()=>yD,minimumImpl:()=>xD,multiplyImpl:()=>CD,negImpl:()=>ED,notEqualImpl:()=>kD,prodImpl:()=>PD,raggedGatherImpl:()=>WD,raggedRangeImpl:()=>KD,raggedTensorToTensorImpl:()=>ZD,rangeImpl:()=>QD,rsqrtImpl:()=>$D,scatterImpl:()=>tO,sigmoidImpl:()=>nO,simpleAbsImpl:()=>yE,sliceImpl:()=>aO,sparseFillEmptyRowsImpl:()=>cO,sparseReshapeImpl:()=>lO,sparseSegmentReductionImpl:()=>uO,sqrtImpl:()=>dO,squaredDifferenceImpl:()=>pO,staticRegexReplaceImpl:()=>hO,stridedSliceImpl:()=>_O,stringNGramsImpl:()=>yO,stringSplitImpl:()=>xO,stringToHashBucketFastImpl:()=>SO,subImpl:()=>CO,tileImpl:()=>EO,topKImpl:()=>kO,transposeImpl:()=>jD,uniqueImpl:()=>AO});wa(`cpu`,()=>new vE,1);var MO=HE(`Elu`,e=>e>=0?e:Math.exp(e)-1),NO={kernelName:`Elu`,backendName:`cpu`,kernelFunc:MO};function PO(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{alpha:a}=r;J([i],`leakyRelu`);let o=C(i.shape),s=n.data.get(i.dataId).values,c=re(`float32`,o);for(let e=0;e<s.length;e++)c[e]=s[e]<0?a*s[e]:s[e];return n.makeTensorInfo(i.shape,`float32`,c)}var FO={kernelName:Xt,backendName:`cpu`,kernelFunc:PO},IO=xE((e,t)=>e<0?t*e:e);function LO(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t;J([r,i],`prelu`);let a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,[s,c]=IO(r.shape,i.shape,a,o,`float32`);return n.makeTensorInfo(c,`float32`,s)}var RO={kernelName:En,backendName:`cpu`,kernelFunc:LO},zO=HE(Pn,e=>Math.max(0,e)),BO={kernelName:Pn,backendName:`cpu`,kernelFunc:zO},VO=HE(Bn,e=>Math.min(Math.max(0,e),6)),HO={kernelName:Bn,backendName:`cpu`,kernelFunc:VO};function UO(e,t,n,r,i){if(n===`linear`)return TE({inputs:{x:t},backend:e});if(n===`relu`)return zO({inputs:{x:t},backend:e});if(n===`elu`)return MO({inputs:{x:t},backend:e});if(n===`relu6`)return VO({inputs:{x:t},backend:e});if(n===`prelu`)return LO({inputs:{x:t,alpha:r},backend:e});if(n===`leakyrelu`)return PO({inputs:{x:t},backend:e,attrs:{alpha:i}});if(n===`sigmoid`)return rO({inputs:{x:t},backend:e});throw Error(`Activation ${n} has not been implemented for the CPU backend.`)}function WO(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{shape:a}=r,o=C(i.shape),s=te(a,o),c=C(s);b(o===c,()=>`The new shape (${s}) has ${c} elements and the old shape (${i.shape}) has ${o} elements. The new shape and old shape must have the same number of elements.`),n.incRef(i.dataId);let l=n.data.get(i.dataId);if(l.complexTensorInfos!=null){let e=l.complexTensorInfos.real,t=l.complexTensorInfos.imag;e.shape=s,t.shape=s}return{dataId:i.dataId,shape:s,dtype:i.dtype}}var GO={kernelName:Fn,backendName:`cpu`,kernelFunc:WO};function KO(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;J([i,a],`matMul`);let c=i.shape.length,l=a.shape.length,u=o?i.shape[c-2]:i.shape[c-1],d=s?a.shape[l-1]:a.shape[l-2],f=o?i.shape[c-1]:i.shape[c-2],p=s?a.shape[l-2]:a.shape[l-1],m=i.shape.slice(0,-2),h=a.shape.slice(0,-2),g=C(m),_=C(h),v=V(i.shape.slice(0,-2),a.shape.slice(0,-2)).concat([f,p]);b(u===d,()=>`Error in matMul: inner shapes (${u}) and (${d}) of Tensors with shapes ${i.shape} and ${a.shape} and transposeA=${o} and transposeB=${s} must match.`);let y=o?[g,u,f]:[g,f,u],x=s?[_,p,d]:[_,d,p],S=WO({inputs:{x:i},backend:n,attrs:{shape:y}}),w=WO({inputs:{x:a},backend:n,attrs:{shape:x}}),T=o?S.shape[1]:S.shape[2],E=o?S.shape[2]:S.shape[1],D=s?w.shape[1]:w.shape[2],ee=Math.max(g,_),te=n.data.get(S.dataId).values,O=n.data.get(w.dataId).values,ne=k(S.shape),re=k(w.shape),[ie,ae,oe]=o?[ne[0],1,ne[1]]:[ne[0],ne[1],1],[se,ce,le]=s?[1,re[1],re[0]]:[re[1],1,re[0]],ue=E*D,de=_o([ee,E,D],S.dtype),fe=de.values,pe=n.blockSize;for(let e=0;e<ee;e++){let t=e%g,n=e%_;for(let r=0;r<E;r+=pe){let i=Math.min(r+pe,E);for(let a=0;a<D;a+=pe){let o=Math.min(a+pe,D);for(let s=0;s<T;s+=pe){let c=Math.min(s+pe,T);for(let l=r;l<i;l++)for(let r=a;r<o;r++){let i=0;for(let e=s;e<c;e++){let a=te[t*ie+l*ae+e*oe],o=O[e*se+r*ce+n*le];i+=a*o}fe[e*ue+(l*D+r)]+=i}}}}}return n.disposeIntermediateTensorInfo(S),n.disposeIntermediateTensorInfo(w),n.makeTensorInfo(v,de.dtype,de.values)}var qO={kernelName:Ze,backendName:`cpu`,kernelFunc:KO};function JO(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r,f,p,m,h=[];f=KO({inputs:{a:i,b:a},attrs:{transposeA:c,transposeB:l},backend:n}),o&&(p=FE({inputs:{a:f,b:o},backend:n}),h.push(f),f=p),u&&(m=UO(n,f,u,s,d),h.push(f),f=m);for(let e of h)n.disposeIntermediateTensorInfo(e);return f}var YO={kernelName:Or,backendName:`cpu`,kernelFunc:JO},XO={kernelName:Le,backendName:`cpu`,kernelFunc:HE(Le,e=>Math.acos(e))},ZO={kernelName:Re,backendName:`cpu`,kernelFunc:HE(Re,e=>Math.acosh(e))};function QO(e){let{inputs:t,backend:n}=e,r=t;J(t,`addN`);let i=r.map(e=>n.data.get(e.dataId).values),a=_o(r[0].shape,r[0].dtype),o=a.values;for(let e=0;e<r.length;e++){let t=i[e];for(let e=0;e<o.length;e++)o[e]+=t[e]}return n.makeTensorInfo(a.shape,a.dtype,a.values)}var $O={kernelName:ze,backendName:`cpu`,kernelFunc:QO};function ek(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;J(i,`all`);let s=O(a,i.shape),c=s,l=ol(c,i.shape.length),u=i;l!=null&&(u=MD({inputs:{x:i},backend:n,attrs:{perm:l}}),c=cl(c.length,i.shape.length)),al(`all`,c,u.shape.length);let[d,f]=rl(u.shape,c),p=C(f),m=be(C(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];n&&=r}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let g=n.makeTensorInfo(d,u.dtype,m);if(o){let e=il(d,s),t=WO({inputs:{x:g},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(g),t}return g}var tk={kernelName:`All`,backendName:`cpu`,kernelFunc:ek};function nk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;J(i,`any`);let s=O(a,i.shape),c=s,l=ol(c,i.shape.length),u=i;l!=null&&(u=MD({inputs:{x:i},backend:n,attrs:{perm:l}}),c=cl(c.length,i.shape.length)),al(`any`,c,u.shape.length);let[d,f]=rl(u.shape,c),p=C(f),m=be(C(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];n||=r}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let g=n.makeTensorInfo(d,u.dtype,m);if(o){let e=il(d,s),t=WO({inputs:{x:g},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(g),t}return g}var rk={kernelName:`Any`,backendName:`cpu`,kernelFunc:nk};function ik(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r;J(i,`argMax`);let o=O(a,i.shape),s=ol(o,i.shape.length),c=i,l=[];s!=null&&(c=MD({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=cl(o.length,c.shape.length)),o=[o[0]],al(`argMax`,o,c.shape.length);let[u,d]=rl(c.shape,o),f=be(C(u),`int32`),p=C(d),m=n.data.get(c.dataId).values;for(let e=0;e<f.length;++e){let t=e*p,n=m[t],r=0;for(let e=0;e<p;++e){let i=m[t+e];i>n&&(n=i,r=e)}f[e]=r}return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(u,`int32`,f)}var ak={kernelName:Be,backendName:`cpu`,kernelFunc:ik};function ok(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r;J(i,`argMin`);let o=O(a,i.shape),s=ol(o,i.shape.length),c=i,l=[];s!=null&&(c=MD({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=cl(o.length,c.shape.length)),o=[o[0]],al(`argMin`,o,c.shape.length);let[u,d]=rl(c.shape,o),f=be(C(u),`int32`),p=C(d),m=n.data.get(c.dataId).values;for(let e=0;e<f.length;++e){let t=e*p,n=m[t],r=0;for(let e=0;e<p;++e){let i=m[t+e];i<n&&(n=i,r=e)}f[e]=r}return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(u,`int32`,f)}var sk={kernelName:Ve,backendName:`cpu`,kernelFunc:ok},ck={kernelName:He,backendName:`cpu`,kernelFunc:HE(He,e=>Math.asin(e))},lk={kernelName:Ue,backendName:`cpu`,kernelFunc:HE(Ue,e=>Math.asinh(e))},uk={kernelName:We,backendName:`cpu`,kernelFunc:HE(We,e=>Math.atan(e))},dk={kernelName:Ke,backendName:`cpu`,kernelFunc:ME(Ke,xE((e,t)=>Math.atan2(e,t)))},fk={kernelName:Ge,backendName:`cpu`,kernelFunc:HE(Ge,e=>Math.atanh(e))};function pk(e,t,n,r,i,a){let o=i.strideHeight,s=i.strideWidth,c=i.dilationHeight,l=i.dilationWidth,u=i.effectiveFilterHeight,d=i.effectiveFilterWidth,f=i.padInfo.top,p=i.padInfo.left,m=a===`max`?-1/0:1/0,h=_o(i.outShape,n),g=h.values,_=i.outShape[1]*i.outShape[2]*i.outShape[3],v=i.outShape[2]*i.outShape[3],y=i.outShape[3];for(let t=0;t<i.batchSize;++t){let n=t*_,h=t*r[0];for(let t=0;t<i.inChannels;++t)for(let _=0;_<i.outHeight;++_){let b=_*o-f,x=Math.max(0,b),S=Math.min(i.inHeight,u+b),C=n+_*v;for(let n=0;n<i.outWidth;++n){let o=n*s-p,u=Math.max(0,o),f=Math.min(i.inWidth,d+o),_=m,v=0,b=0;for(let n=x;n<S;n+=c){let i=h+n*r[1];for(let n=u;n<f;n+=l){let o=e[i+n*r[2]+t];a===`max`&&o>_?_=o:a===`avg`&&(v+=o,b++)}if(isNaN(_))break}let w=C+n*y+t;g[w]=a===`avg`?v/b:_}}}return h}function mk(e,t,n,r,i=!1,a=!1){let o=_o(r.outShape,`int32`),s=r.strideHeight,c=r.strideWidth,l=r.dilationHeight,u=r.dilationWidth,d=r.effectiveFilterHeight,f=r.effectiveFilterWidth,p=r.padInfo.top,m=r.padInfo.left,h=_o(t,n,e);for(let e=0;e<r.batchSize;++e)for(let t=0;t<r.inChannels;++t)for(let n=0;n<r.outHeight;++n){let g=n*s-p,_=g;for(;_<0;)_+=l;let v=Math.min(r.inHeight,d+g);for(let s=0;s<r.outWidth;++s){let d=s*c-m,p=d;for(;p<0;)p+=u;let y=Math.min(r.inWidth,f+d),b=-1/0,x=-1;for(let n=_;n<v;n+=l){let o=n-g;for(let s=p;s<y;s+=u){let c=s-d,l=h.get(e,n,s,t);l>b&&(b=l,x=i?a?((e*r.inHeight+n)*r.inWidth+s)*r.inChannels+t:(n*r.inWidth+s)*r.inChannels+t:o*f+c)}}o.set(x,e,n,s,t)}}return o}function hk(e,t,n,r,i,a){let o=i.strideDepth,s=i.strideHeight,c=i.strideWidth,l=i.dilationDepth,u=i.dilationHeight,d=i.dilationWidth,f=i.effectiveFilterDepth,p=i.effectiveFilterHeight,m=i.effectiveFilterWidth,h=i.padInfo.front,g=i.padInfo.top,_=i.padInfo.left,v=a===`max`?-1/0:1/0,y=_o(i.outShape,n),b=y.values,x=i.outShape[1]*i.outShape[2]*i.outShape[3]*i.outShape[4],S=i.outShape[2]*i.outShape[3]*i.outShape[4],C=i.outShape[3]*i.outShape[4],w=i.outShape[4];for(let t=0;t<i.batchSize;++t){let n=t*x,y=t*r[0];for(let t=0;t<i.inChannels;++t)for(let x=0;x<i.outDepth;++x){let T=x*o-h,E=T;for(;E<0;)E+=l;let D=Math.min(i.inDepth,f+T),ee=n+x*S;for(let n=0;n<i.outHeight;++n){let o=n*s-g,f=o;for(;f<0;)f+=u;let h=Math.min(i.inHeight,p+o),x=ee+n*C;for(let n=0;n<i.outWidth;++n){let o=n*c-_,s=o;for(;s<0;)s+=d;let p=Math.min(i.inWidth,m+o),g=x+n*w,S=v,C=0,T=0;for(let n=E;n<D;n+=l){let i=y+n*r[1];for(let n=f;n<h;n+=u){let o=i+n*r[2];for(let n=s;n<p;n+=d){let i=e[o+n*r[3]+t];if(a===`max`&&i>S?S=i:a===`avg`&&(C+=i,T++),isNaN(S))break}if(isNaN(S))break}if(isNaN(S))break}let ee=g+t;b[ee]=a===`avg`?C/Math.max(T,1):S}}}}return y}function gk(e,t){let n=_o(t.outShape,`int32`),r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,o=t.dilationDepth,s=t.dilationHeight,c=t.dilationWidth,l=t.effectiveFilterDepth,u=t.effectiveFilterHeight,d=t.effectiveFilterWidth,f=t.padInfo.front,p=t.padInfo.top,m=t.padInfo.left;for(let h=0;h<t.batchSize;++h)for(let g=0;g<t.inChannels;++g)for(let _=0;_<t.outDepth;++_){let v=_*r-f,y=v;for(;y<0;)y+=o;let b=Math.min(t.inDepth,l+v);for(let r=0;r<t.outHeight;++r){let l=r*i-p,f=l;for(;f<0;)f+=s;let x=Math.min(t.inHeight,u+l);for(let i=0;i<t.outWidth;++i){let p=i*a-m,S=p;for(;S<0;)S+=c;let C=Math.min(t.inWidth,d+p),w=-1/0,T=-1;for(let t=y;t<b;t+=o){let n=t-v;for(let r=f;r<x;r+=s){let i=r-l;for(let a=S;a<C;a+=c){let o=a-p,s=e.get(h,t,r,a,g);s>=w&&(w=s,T=n*u*d+i*u+o)}}}n.set(T,h,_,r,i,g)}}}return n}function _k(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;J(i,`avgPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;b(ms(o,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=es(i.shape,a,o,1,s,c),u;if(l.filterWidth===1&&l.filterHeight===1&&w(l.inShape,l.outShape))u=TE({inputs:{x:i},backend:n});else{let e=n.data.get(i.dataId).values,t=k(i.shape),r=pk(e,i.shape,i.dtype,t,l,`avg`);u=n.makeTensorInfo(l.outShape,i.dtype,r.values)}return u}var vk={kernelName:qe,backendName:`cpu`,kernelFunc:_k};function yk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r;J(i,`avgPool3d`);let u=ts(i.shape,a,o,1,s,c,l),d=n.data.get(i.dataId).values,f=hk(d,i.shape,i.dtype,k(i.shape),u,`avg`);return n.makeTensorInfo(f.shape,`float32`,f.values)}var bk={kernelName:Ye,backendName:`cpu`,kernelFunc:yk};function xk(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r;J([i,a],`avgPool3DGrad`);let u=ts(a.shape,o,s,1,c,l),d=u.strideDepth,f=u.strideHeight,p=u.strideWidth,m=u.filterDepth,h=u.filterHeight,g=u.filterWidth,_=u.dilationDepth,v=u.dilationHeight,y=u.dilationWidth,b=u.effectiveFilterDepth,x=u.effectiveFilterHeight,S=u.effectiveFilterWidth,C=b-1-u.padInfo.front,w=S-1-u.padInfo.left,T=x-1-u.padInfo.top,E=_o(a.shape,`float32`),D=1/(m*h*g),ee=n.bufferSync(i);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inDepth;++n)for(let r=0;r<u.inHeight;++r)for(let i=0;i<u.inWidth;++i){let a=n-C,o=r-T,s=i-w,c=0;for(let n=0;n<b;n+=_){let r=(a+n)/d;if(!(r<0||r>=u.outDepth||Math.floor(r)!==r))for(let n=0;n<x;n+=v){let i=(o+n)/f;if(!(i<0||i>=u.outHeight||Math.floor(i)!==i))for(let n=0;n<S;n+=y){let a=(s+n)/p;if(a<0||a>=u.outWidth||Math.floor(a)!==a)continue;let o=ee.get(e,r,i,a,t);c+=o}}}E.set(c*D,e,n,r,i,t)}return n.makeTensorInfo(E.shape,E.dtype,E.values)}var Sk={kernelName:Xe,backendName:`cpu`,kernelFunc:xk};function Ck(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a;J([i,a],`avgPoolGrad`);let{filterSize:s,strides:c,pad:l}=r,u=es(o.shape,s,c,1,l),d=u.strideHeight,f=u.strideWidth,p=u.filterHeight,m=u.filterWidth,h=u.dilationHeight,g=u.dilationWidth,_=u.effectiveFilterHeight,v=u.effectiveFilterWidth,y=v-1-u.padInfo.left,b=_-1-u.padInfo.top,x=_o(o.shape,`float32`),S=1/(p*m),C=n.data.get(i.dataId).values,w=_o(i.shape,`float32`,C);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inHeight;++n)for(let r=0;r<u.inWidth;++r){let i=n-b,a=r-y,o=0;for(let n=0;n<_;n+=h){let r=(i+n)/d;if(!(r<0||r>=u.outHeight||Math.floor(r)!==r))for(let n=0;n<v;n+=g){let i=(a+n)/f;if(i<0||i>=u.outWidth||Math.floor(i)!==i)continue;let s=w.get(e,r,i,t);o+=s}}x.set(o*S,e,n,r,t)}return n.makeTensorInfo(x.shape,x.dtype,x.values)}var wk={kernelName:Je,backendName:`cpu`,kernelFunc:Ck};function Tk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,scale:a,offset:o,mean:s,variance:c}=t;b(s.shape.length===c.shape.length,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),b(o==null||s.shape.length===o.shape.length,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),b(a==null||s.shape.length===a.shape.length,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`),J([i,s,c,a,o],`batchNorm`);let{varianceEpsilon:l}=r;l??=.001;let u=n.data.get(i.dataId).values,d=n.data.get(s.dataId).values,f=n.data.get(c.dataId).values,p=a?n.data.get(a.dataId).values:new Float32Array([1]),m=o?n.data.get(o.dataId).values:new Float32Array([0]),h=new Float32Array(u.length),g=m.length,_=p.length,v=f.length,y=d.length,x=0,S=0,C=0,w=0;for(let e=0;e<u.length;++e)h[e]=m[x++]+(u[e]-d[S++])*p[C++]/Math.sqrt(f[w++]+l),x>=g&&(x=0),S>=y&&(S=0),C>=_&&(C=0),w>=v&&(w=0);return n.makeTensorInfo(i.shape,i.dtype,h)}var Ek={kernelName:zt,backendName:`cpu`,kernelFunc:Tk};function Dk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r;J([i],`batchToSpaceND`);let s=a.reduce((e,t)=>e*t),c=Dh(i.shape,a,s),l=Oh(c.length,a.length),u=kh(i.shape,a,s),d=Ah(o,a.length),f=jh(u,o,a.length),p=WO({inputs:{x:i},backend:n,attrs:{shape:c}}),m=MD({inputs:{x:p},backend:n,attrs:{perm:l}}),h=WO({inputs:{x:m},backend:n,attrs:{shape:u}}),g=oO({inputs:{x:h},backend:n,attrs:{begin:d,size:f}});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),g}var Ok={kernelName:Qe,backendName:`cpu`,kernelFunc:Dk};function kk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o}=r,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,l=LE(s,c,a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,l)}var Ak={kernelName:$e,backendName:`cpu`,kernelFunc:kk};function jk(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t,a=n.data.get(r.dataId).values,o=n.data.get(i.dataId).values,s=V(Array.from(a),Array.from(o));return n.makeTensorInfo([s.length],`int32`,Int32Array.from(s))}var Mk={kernelName:nt,backendName:`cpu`,kernelFunc:jk},Nk={kernelName:at,backendName:`cpu`,kernelFunc:HE(at,(e,t)=>{let n=t;return e>n.clipValueMax?n.clipValueMax:e<n.clipValueMin?n.clipValueMin:e})},Pk={kernelName:st,backendName:`cpu`,kernelFunc:e=>{let{x:t}=e.inputs,n=e.backend,r=new Float32Array(C(t.shape)),i=n.data.get(t.dataId),a=i.complexTensorInfos.real,o=i.complexTensorInfos.imag,s=n.data.get(a.dataId).values,c=n.data.get(o.dataId).values;for(let e=0;e<s.length;e++){let t=s[e],n=c[e];r[e]=Math.hypot(t,n)}return n.makeOutput(r,t.shape,`float32`)}};function Fk(e){let{inputs:t,backend:n}=e,{input:r}=t,i=n.data.get(r.dataId).complexTensorInfos.imag,a=n.data.get(i.dataId).values;return n.makeTensorInfo(i.shape,i.dtype,a)}var Ik={kernelName:Kt,backendName:`cpu`,kernelFunc:Fk};function Lk(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r,a=O(i,t[0].shape)[0];vh(t.map(e=>e.shape),a);let o=yh(t.map(e=>e.shape),a);if(C(o)===0)return n.makeTensorInfo(o,t[0].dtype,[]);let s=t.filter(e=>C(e.shape)>0);if(s.length===1)return TE({inputs:{x:s[0]},backend:n});if(s[0].dtype===`complex64`){let e=s.map(e=>DE({inputs:{input:e},backend:n})),t=s.map(e=>Fk({inputs:{input:e},backend:n})),r=Lk({inputs:e,backend:n,attrs:{axis:a}}),i=Lk({inputs:t,backend:n,attrs:{axis:a}}),o=SE({inputs:{real:r,imag:i},backend:n});return e.forEach(e=>n.disposeIntermediateTensorInfo(e)),t.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.disposeIntermediateTensorInfo(r),n.disposeIntermediateTensorInfo(i),o}let c=s.map(e=>{let t=[-1,C(e.shape.slice(a))];return WO({inputs:{x:e},backend:n,attrs:{shape:t}})}),l=c.map(e=>({vals:n.data.get(e.dataId).values,shape:e.shape}));o=yh(c.map(e=>e.shape),1);let u=c[0].shape[0]===1,d=KE(l,o,t[0].dtype,u),f=yh(s.map(e=>e.shape),a),p=n.makeTensorInfo(f,t[0].dtype,d);return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}var Rk={kernelName:ct,backendName:`cpu`,kernelFunc:Lk};function zk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dataFormat:c,dilations:l,dimRoundingMode:u}=r;J([i,a],`conv2d`);let d=gs(c),f=ns(i.shape,a.shape,o,l,s,u,!1,d),p=f.filterHeight,m=f.filterWidth,h=f.dilationHeight,g=f.dilationWidth,_=f.padInfo.left,v=f.padInfo.top,y=f.dataFormat===`channelsLast`,b=new Oi(f.outShape,i.dtype),x=k(i.shape),S=k(a.shape),C=x[0],w=y?x[1]:x[2],T=y?x[2]:1,E=y?1:x[1],D=b.strides[0],ee=y?b.strides[1]:b.strides[2],te=y?b.strides[2]:1,O=y?1:b.strides[1],ne=n.data.get(i.dataId).values,re=n.data.get(a.dataId).values,ie=b.values;for(let e=0;e<f.batchSize;++e){let t=e*C,n=e*D;for(let e=0;e<f.outHeight;++e){let r=n+e*ee,i=e*f.strideHeight-v;for(let e=0;e<p;++e){let n=i+e*h;if(n<0||n>=f.inHeight)continue;let a=e*S[0],o=t+n*w;for(let e=0;e<f.outWidth;++e){let t=r+e*te,n=e*f.strideWidth-_;for(let e=0;e<m;++e){let r=n+e*g;if(r<0||r>=f.inWidth)continue;let i=a+e*S[1],s=o+r*T,c=i;for(let e=0;e<f.inChannels;++e){let n=ne[s+e*E];for(let e=0;e<f.outChannels;++e)ie[t+e*O]+=n*re[c+e];c+=f.outChannels}}}}}}return n.makeTensorInfo(b.shape,b.dtype,ie)}var Bk={kernelName:lt,backendName:`cpu`,kernelFunc:zk};function Vk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,filterShape:u}=r;J([i,a],`conv2dBackpropFilter`);let d=gs(c),f=ns(i.shape,u,o,1,s,l,!1,d),{strideHeight:p,strideWidth:m,filterHeight:h,filterWidth:g}=f,_=f.dataFormat===`channelsLast`,v=new Oi(f.filterShape,`float32`),y=f.padInfo.left,b=f.padInfo.top,x=n.data.get(i.dataId).values,S=n.data.get(a.dataId).values,C=new Oi(i.shape,i.dtype,x),w=new Oi(a.shape,a.dtype,S);for(let e=0;e<h;++e){let t=Math.max(0,Math.ceil((b-e)/p)),n=Math.min(f.outHeight,(f.inHeight+b-e)/p);for(let r=0;r<g;++r){let i=Math.max(0,Math.ceil((y-r)/m)),a=Math.min(f.outWidth,(f.inWidth+y-r)/m);for(let o=0;o<f.inChannels;++o)for(let s=0;s<f.outChannels;++s){let c=0;for(let l=0;l<f.batchSize;++l)for(let u=t;u<n;++u){let t=e+u*p-b;for(let e=i;e<a;++e){let n=r+e*m-y;c+=_?C.get(l,t,n,o)*w.get(l,u,e,s):C.get(l,o,t,n)*w.get(l,s,u,e)}}v.set(c,e,r,o,s)}}}return n.makeTensorInfo(v.shape,v.dtype,v.values)}var Hk={kernelName:ut,backendName:`cpu`,kernelFunc:Vk};function Uk(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{inputShape:o,strides:s,pad:c,dataFormat:l,dimRoundingMode:u}=r;J([i,a],`conv2dBackpropInput`);let d=k(a.shape),f=k(i.shape),p=gs(l),m=ns(o,a.shape,s,1,c,u,!1,p),h=new Oi(m.inShape,`float32`),g=h.values,_=n.data.get(i.dataId).values,v=n.data.get(a.dataId).values,[y,b,x]=d,{batchSize:S,filterHeight:C,filterWidth:w,inChannels:T,inHeight:E,inWidth:D,outChannels:ee,outHeight:te,outWidth:O,strideHeight:ne,strideWidth:re}=m;p=m.dataFormat;let ie=C-1-m.padInfo.top,ae=w-1-m.padInfo.left,oe=p===`channelsLast`,se=h.strides[0],ce=oe?h.strides[1]:h.strides[2],le=oe?h.strides[2]:1,ue=oe?1:h.strides[1],de=f[0],fe=oe?f[1]:f[2],pe=oe?f[2]:1,me=oe?1:f[1];for(let e=0;e<S;++e)for(let t=0;t<T;++t)for(let n=0;n<E;++n){let r=n-ie,i=Math.max(0,Math.ceil(r/ne)),a=Math.min(te,(C+r)/ne);for(let o=0;o<D;++o){let s=o-ae,c=Math.max(0,Math.ceil(s/re)),l=Math.min(O,(w+s)/re),u=0;for(let n=i;n<a;++n){let i=n*ne-r;for(let r=c;r<l;++r){let a=r*re-s,o=de*e+fe*n+pe*r,c=y*(C-1-i)+b*(w-1-a)+x*t;for(let e=0;e<ee;++e){let t=_[o+me*e],n=v[c+e];u+=t*n}}}let d=se*e+ce*n+le*o+ue*t;g[d]=u}}return n.makeTensorInfo(h.shape,h.dtype,h.values)}var Wk={kernelName:dt,backendName:`cpu`,kernelFunc:Uk};function Gk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r;J([i,a],`conv3d`);let l=rs(i.shape,a.shape,o,c,s),{filterDepth:u,filterHeight:d,filterWidth:f,dilationDepth:p,dilationHeight:m,dilationWidth:h,padInfo:g}=l,_=g.front,v=g.left,y=g.top,b=new Oi(l.outShape,i.dtype),x=n.data.get(i.dataId).values,S=n.data.get(a.dataId).values,C=b.values,w=k(i.shape),T=k(a.shape);for(let e=0;e<l.batchSize;++e){let t=e*w[0],n=e*b.strides[0];for(let e=0;e<l.outDepth;++e){let r=n+e*b.strides[1],i=e*l.strideDepth-_;for(let e=0;e<u;++e){let n=i+e*p;if(n<0||n>=l.inDepth)continue;let a=e*T[0],o=t+n*w[1];for(let e=0;e<l.outHeight;++e){let t=r+e*b.strides[2],n=e*l.strideHeight-y;for(let e=0;e<d;++e){let r=n+e*m;if(r<0||r>=l.inHeight)continue;let i=a+e*T[1],s=o+r*w[2];for(let e=0;e<l.outWidth;++e){let n=t+e*l.outChannels,r=e*l.strideWidth-v;for(let e=0;e<f;++e){let t=r+e*h;if(t<0||t>=l.inWidth)continue;let a=i+e*T[2],o=s+t*l.inChannels,c=a;for(let e=0;e<l.inChannels;++e){let t=x[o+e];for(let e=0;e<l.outChannels;++e)C[n+e]+=t*S[c+e];c+=l.outChannels}}}}}}}}return n.makeTensorInfo(b.shape,b.dtype,b.values)}var Kk={kernelName:ft,backendName:`cpu`,kernelFunc:Gk};function qk(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r;J([i,a],`conv3dBackpropFilterV2`);let l=k(i.shape),u=k(a.shape),d=rs(i.shape,c,o,1,s),f=d.strideDepth,p=d.strideHeight,m=d.strideWidth,h=d.filterDepth,g=d.filterHeight,_=d.filterWidth,v=new Oi(d.filterShape,`float32`),y=v.values,[b,x,S,C]=v.strides,w=n.data.get(a.dataId).values,[T,E,D,ee]=u,te=n.data.get(i.dataId).values,[O,ne,re,ie]=l,ae=d.padInfo.front,oe=d.padInfo.left,se=d.padInfo.top;for(let e=0;e<h;++e){let t=Math.max(0,Math.ceil((ae-e)/f)),n=Math.min(d.outDepth,(d.inDepth+ae-e)/f),r=e*b;for(let i=0;i<g;++i){let a=Math.max(0,Math.ceil((se-i)/p)),o=Math.min(d.outHeight,(d.inHeight+se-i)/p),s=i*x+r;for(let r=0;r<_;++r){let c=Math.max(0,Math.ceil((oe-r)/m)),l=Math.min(d.outWidth,(d.inWidth+oe-r)/m),u=r*S+s;for(let s=0;s<d.inChannels;++s){let h=s*C+u;for(let u=0;u<d.outChannels;++u){let g=0;for(let h=0;h<d.batchSize;++h){let d=h*O,_=h*T;for(let h=t;h<n;++h){let t=(e+h*f-ae)*ne+d,n=h*E+_;for(let e=a;e<o;++e){let a=(i+e*p-se)*re+t,o=e*D+n;for(let e=c;e<l;++e){let t=(r+e*m-oe)*ie+a,n=e*ee+o;g+=te[t+s]*w[n+u]}}}}y[h+u]=g}}}}}return n.makeTensorInfo(v.shape,v.dtype,v.values)}var Jk={kernelName:pt,backendName:`cpu`,kernelFunc:qk};function Yk(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{pad:o,strides:s,inputShape:c}=r;J([i],`conv3dBackpropInputV2`);let l=k(i.shape),u=k(a.shape),d=rs(c,a.shape,s,1,o),f=new Oi(d.inShape,`float32`),p=f.values,[m,h,g,_]=f.strides,v=n.data.get(i.dataId).values,[y,b,x,S]=l,C=n.data.get(a.dataId).values,[w,T,E,D]=u,{batchSize:ee,filterDepth:te,filterHeight:O,filterWidth:ne,inChannels:re,inDepth:ie,inHeight:ae,inWidth:oe,outChannels:se,outDepth:ce,outHeight:le,outWidth:ue,strideDepth:de,strideHeight:fe,strideWidth:pe}=d,me=te-1-d.padInfo.front,he=O-1-d.padInfo.top,ge=ne-1-d.padInfo.left;for(let e=0;e<ee;++e)for(let t=0;t<re;++t)for(let n=0;n<ie;++n){let r=n-me,i=Math.max(0,Math.ceil(r/de)),a=Math.min(ce,(te+r)/de);for(let o=0;o<ae;++o){let s=o-he,c=Math.max(0,Math.ceil(s/fe)),l=Math.min(le,(O+s)/fe);for(let u=0;u<oe;++u){let d=u-ge,f=Math.max(0,Math.ceil(d/pe)),ee=Math.min(ue,(ne+d)/pe),re=0;for(let n=i;n<a;++n){let i=n*de-r;for(let r=c;r<l;++r){let a=r*fe-s;for(let o=f;o<ee;++o){let s=o*pe-d,c=y*e+b*n+x*r+S*o,l=w*(te-1-i)+T*(O-1-a)+E*(ne-1-s)+D*t;for(let e=0;e<se;++e){let t=v[c+e],n=C[l+e];re+=t*n}}}}p[m*e+h*n+g*o+_*u+t]=re}}}return n.makeTensorInfo(f.shape,f.dtype,f.values)}var Xk={kernelName:mt,backendName:`cpu`,kernelFunc:Yk},Zk={kernelName:`Cos`,backendName:`cpu`,kernelFunc:HE(`Cos`,e=>Math.cos(e))},Qk={kernelName:ht,backendName:`cpu`,kernelFunc:HE(ht,e=>Math.cosh(e))};function $k(e){let{inputs:t,backend:n,attrs:r}=e,{image:i,boxes:a,boxInd:o}=t,{cropSize:s,method:c,extrapolationValue:l}=r,[u,d,f,p]=i.shape,m=a.shape[0],[h,g]=s,_=_o([m,h,g,p],`float32`),v=n.data.get(a.dataId).values,y=n.data.get(o.dataId).values,b=n.data.get(i.dataId).values,x=k(i.shape),S=k(_.shape);for(let e=0;e<m;e++){let t=e*4,n=v[t],r=v[t+1],i=v[t+2],a=v[t+3],o=y[e];if(o>=u)continue;let s=h>1?(i-n)*(d-1)/(h-1):0,m=g>1?(a-r)*(f-1)/(g-1):0;for(let t=0;t<h;t++){let u=h>1?n*(d-1)+t*s:.5*(n+i)*(d-1);if(u<0||u>d-1){for(let n=0;n<g;n++)for(let r=0;r<p;r++){let i=r+n*S[2]+t*S[1]+e*S[0];_.values[i]=l}continue}if(c===`bilinear`){let n=Math.floor(u),i=Math.ceil(u),s=u-n;for(let c=0;c<g;c++){let u=g>1?r*(f-1)+c*m:.5*(r+a)*(f-1);if(u<0||u>f-1){for(let n=0;n<p;n++){let r=n+c*S[2]+t*S[1]+e*S[0];_.values[r]=l}continue}let d=Math.floor(u),h=Math.ceil(u),v=u-d;for(let r=0;r<p;r++){let a=r+d*x[2]+n*x[1]+o*x[0],l=b[a];a=r+h*x[2]+n*x[1]+o*x[0];let u=b[a];a=r+d*x[2]+i*x[1]+o*x[0];let f=b[a];a=r+h*x[2]+i*x[1]+o*x[0];let p=b[a],m=l+(u-l)*v,g=f+(p-f)*v;a=r+c*S[2]+t*S[1]+e*S[0],_.values[a]=m+(g-m)*s}}}else for(let n=0;n<g;++n){let i=g>1?r*(f-1)+n*m:.5*(r+a)*(f-1);if(i<0||i>f-1){for(let r=0;r<p;r++){let i=r+n*S[2]+t*S[1]+e*S[0];_.values[i]=l}continue}let s=Math.round(i),c=Math.round(u);for(let r=0;r<p;r++){let i=r+s*x[2]+c*x[1]+o*x[0],a=r+n*S[2]+t*S[1]+e*S[0];_.values[a]=b[i]}}}}return n.makeTensorInfo(_.shape,_.dtype,_.values)}var eA={kernelName:vt,backendName:`cpu`,kernelFunc:$k};function tA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;J(i,`cumprod`);let c=ol([a],i.shape.length),l=i;c!=null&&(l=MD({inputs:{x:i},backend:n,attrs:{perm:c}}));let u=cl(1,i.shape.length)[0];if(u!==l.shape.length-1)throw Error(`backend.cumprod in CPU expects an inner-most axis=${l.shape.length-1} but got axis=${u}`);let d=Vi(l.dtype,`int32`),f=ye(C(l.shape),d),p=n.data.get(l.dataId).values,m=l.shape[l.shape.length-1],h=s?(e,t)=>e+m-t-1:(e,t)=>e+t;for(let e=0;e<p.length;e+=m)for(let t=0;t<m;t++){let n=h(e,t);if(t===0)f[n]=o?1:p[n];else{let r=h(e,t-1);f[n]=o?p[r]*f[r]:p[n]*f[r]}}let g=n.makeTensorInfo(l.shape,d,f);if(c!=null){let e=sl(c),t=MD({inputs:{x:g},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(g),n.disposeIntermediateTensorInfo(l),t}return g}var nA={kernelName:gt,backendName:`cpu`,kernelFunc:tA};function rA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;J(i,`cumsum`);let c=ol([a],i.shape.length),l=i;c!=null&&(l=MD({inputs:{x:i},backend:n,attrs:{perm:c}}));let u=cl(1,i.shape.length)[0];if(u!==l.shape.length-1)throw Error(`backend.cumsum in CPU expects an inner-most axis=${l.shape.length-1} but got axis=${u}`);let d=Vi(l.dtype,`int32`),f=be(C(l.shape),d),p=n.data.get(l.dataId).values,m=l.shape[l.shape.length-1],h=s?(e,t)=>e+m-t-1:(e,t)=>e+t;for(let e=0;e<p.length;e+=m)for(let t=0;t<m;t++){let n=h(e,t);if(t===0)f[n]=o?0:p[n];else{let r=h(e,t-1);f[n]=o?p[r]+f[r]:p[n]+f[r]}}let g=n.makeTensorInfo(l.shape,d,f);if(c!=null){let e=sl(c),t=MD({inputs:{x:g},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(g),n.disposeIntermediateTensorInfo(l),t}return g}var iA={kernelName:_t,backendName:`cpu`,kernelFunc:rA};function aA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o,binaryOutput:s}=r;if(i.shape.length===1){let e=n.data.get(i.dataId).values,t=n.data.get(a.dataId).values,r=LE(e,t,a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,r)}if(i.shape.length===2){let e=RE(n.bufferSync(i),n.bufferSync(a),o,s);return n.makeTensorInfo(e.shape,a.dtype,e.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${i.shape.length}.`)}var oA={kernelName:yt,backendName:`cpu`,kernelFunc:aA};function sA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockSize:a,dataFormat:o}=r;b(o===`NHWC`,()=>`Only NHWC dataFormat supported on CPU for depthToSpace. Got ${o}`);let s=i.shape[0],c=i.shape[1],l=i.shape[2],u=i.shape[3],d=c*a,f=l*a,p=u/(a*a),m=n.data.get(i.dataId).values,h=new Float32Array(s*d*f*p),g=0;for(let e=0;e<s;++e)for(let t=0;t<d;++t){let n=Math.floor(t/a),r=t%a;for(let t=0;t<f;++t){let i=Math.floor(t/a),o=t%a,s=(r*a+o)*p;for(let t=0;t<p;++t){let r=t+s+u*(i+l*(n+c*e));h[g++]=m[r]}}}return n.makeTensorInfo([s,d,f,p],i.dtype,h)}var cA={kernelName:bt,backendName:`cpu`,kernelFunc:sA};function lA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c,dimRoundingMode:l}=r;J([i,a],`depthwiseConv2DNative`);let u=k(i.shape),d=k(a.shape),f=c;f??=[1,1],b(ms(o,f),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${o} and dilations '${f}'`);let p=ns(i.shape,a.shape,o,f,s,l,!0),{filterHeight:m,filterWidth:h,dilationHeight:g,dilationWidth:_,padInfo:v}=p,y=v.left,x=v.top,S=p.outChannels/p.inChannels,C=new Oi(p.outShape,i.dtype),w=n.data.get(i.dataId).values,T=n.data.get(a.dataId).values,E=C.values;for(let e=0;e<p.batchSize;++e){let t=e*u[0],n=e*C.strides[0];for(let e=0;e<p.outHeight;++e){let r=n+e*C.strides[1],i=e*p.strideHeight-x;for(let e=0;e<m;++e){let n=i+e*g;if(n<0||n>=p.inHeight)continue;let a=e*d[0],o=t+n*u[1];for(let e=0;e<p.outWidth;++e){let t=r+e*C.strides[2],n=e*p.strideWidth-y;for(let e=0;e<h;++e){let r=n+e*_;if(r<0||r>=p.inWidth)continue;let i=a+e*d[1],s=o+r*p.inChannels,c=t,l=i;for(let e=0;e<p.inChannels;++e){let t=w[s+e];for(let e=0;e<S;++e)E[c+e]+=t*T[l+e];c+=S,l+=S}}}}}}return n.makeTensorInfo(C.shape,C.dtype,C.values)}var uA={kernelName:xt,backendName:`cpu`,kernelFunc:lA};function dA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,filterShape:u}=r;J([i,a],`depthwiseConv2dNativeBackpropFilter`);let d=ns(i.shape,u,o,s,c,l,!0),{strideHeight:f,strideWidth:p,filterHeight:m,filterWidth:h}=d,g=new Oi(d.filterShape,`float32`),_=d.padInfo.left,v=d.padInfo.top,y=d.outChannels/d.inChannels,b=n.data.get(i.dataId).values,x=new Oi(i.shape,i.dtype,b),S=n.data.get(a.dataId).values,C=new Oi(a.shape,a.dtype,S);for(let e=0;e<m;++e){let t=Math.max(0,Math.ceil((v-e)/f)),n=Math.min(d.outHeight,(d.inHeight+v-e)/f);for(let r=0;r<h;++r){let i=Math.max(0,Math.ceil((_-r)/p)),a=Math.min(d.outWidth,(d.inWidth+_-r)/p);for(let o=0;o<d.outChannels;++o){let s=Math.trunc(o/y),c=o%y,l=0;for(let c=0;c<d.batchSize;++c)for(let u=t;u<n;++u){let t=e+u*f-v;for(let e=i;e<a;++e){let n=r+e*p-_;l+=x.get(c,t,n,s)*C.get(c,u,e,o)}}g.set(l,e,r,s,c)}}}return n.makeTensorInfo(g.shape,g.dtype,g.values)}var fA={kernelName:St,backendName:`cpu`,kernelFunc:dA};function pA(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,inputShape:u}=r;J([i,a],`depthwiseConv2DNativeBackpropInput`);let d=k(i.shape),f=k(a.shape),p=ns(u,a.shape,o,s,c,l,!0),m=new Oi(p.inShape,`float32`),h=m.values,[g,_,v]=m.strides,y=n.data.get(i.dataId).values,[b,x,S]=d,C=n.data.get(a.dataId).values,[w,T,E]=f,{batchSize:D,filterHeight:ee,filterWidth:te,inChannels:O,inHeight:ne,inWidth:re,outChannels:ie,outHeight:ae,outWidth:oe,strideHeight:se,strideWidth:ce}=p,le=ee-1-p.padInfo.top,ue=te-1-p.padInfo.left,de=ie/O;for(let e=0;e<D;++e)for(let t=0;t<O;++t)for(let n=0;n<ne;++n){let r=n-le,i=Math.max(0,Math.ceil(r/se)),a=Math.min(ae,(ee+r)/se);for(let o=0;o<re;++o){let s=o-ue,c=Math.max(0,Math.ceil(s/ce)),l=Math.min(oe,(te+s)/ce),u=0;for(let n=i;n<a;++n){let i=n*se-r;for(let r=c;r<l;++r){let a=r*ce-s,o=b*e+x*n+S*r,c=w*(ee-1-i)+T*(te-1-a)+E*t;for(let e=0;e<de;++e){let n=y[o+(t*de+e)],r=C[c+e];u+=n*r}}}h[g*e+_*n+v*o+t]=u}}return n.makeTensorInfo(m.shape,m.dtype,m.values)}var mA={kernelName:Ct,backendName:`cpu`,kernelFunc:pA};function hA(e){let{inputs:t,backend:n}=e,{x:r}=t,i=C(r.shape),a=n.data.get(r.dataId).values,o=_o([i,i],r.dtype),s=o.values;for(let e=0;e<a.length;e++)s[e*i+e]=a[e];let c=[...r.shape,...r.shape];return n.makeTensorInfo(c,o.dtype,o.values)}var gA={kernelName:wt,backendName:`cpu`,kernelFunc:hA},_A={kernelName:Tt,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,filter:i}=e,{strides:a,pad:o,dilations:s}=n,c=t,l=c.data.get(r.dataId).values,u=r.shape.length,d=c.data.get(i.dataId).values,f=i.shape.length,{batchSize:p,inHeight:m,inWidth:h,inChannels:g,outHeight:_,outWidth:v,padInfo:y,strideHeight:b,strideWidth:x,filterHeight:S,filterWidth:w,dilationHeight:T,dilationWidth:E,outShape:D}=$o(r.shape,i.shape,a,o,`NHWC`,s),ee=C(D),te=D.length,O=ie(r.dtype,ee);for(let e=0;e<p;++e)for(let t=0;t<_;++t){let n=t*b-y.top;for(let a=0;a<v;++a){let o=a*x-y.left;for(let s=0;s<g;++s){let c=-(2**53-1);for(let t=0;t<S;++t){let a=n+t*T;if(a>=0&&a<m)for(let n=0;n<w;++n){let p=o+n*E;if(p>=0&&p<h){let o=Ce([e,a,p,s],u,k(r.shape)),m=Ce([t,n,s],f,k(i.shape)),h=l[o]+d[m];h>c&&(c=h)}}}let p=Ce([e,t,a,s],te,k(D));O[p]=c}}}return{dataId:c.write(ci(O,r.dtype),D,r.dtype),shape:D,dtype:r.dtype}}},vA={kernelName:Dt,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,filter:i,dy:a}=e,{strides:o,pad:s,dilations:c}=n,l=t,u=_e(r.shape,l.data.get(r.dataId).values),d=_e(i.shape,l.data.get(i.dataId).values),{batchSize:f,inHeight:p,inWidth:m,inChannels:h,outHeight:g,outWidth:_,padInfo:v,strideHeight:y,strideWidth:x,filterHeight:S,filterWidth:C,dilationHeight:w,dilationWidth:T,outShape:E}=$o(r.shape,i.shape,o,s,`NHWC`,c);b(a.rank===E.length,()=>`Error in ${Dt}, dy must have the same rank as output ${E.length}, but got ${a.rank}`);let D=_e(E,l.data.get(a.dataId).values),ee=xe(i.shape,i.dtype);for(let e=0;e<f;++e)for(let t=0;t<g;++t){let n=t*y-v.top;for(let r=0;r<_;++r){let i=r*x-v.left;for(let a=0;a<h;++a){let o=-(2**53-1),s=0,c=0;for(let t=0;t<S;++t){let r=n+t*w;if(r>=0&&r<p)for(let n=0;n<C;++n){let l=i+n*T;if(l>=0&&l<m){let i=u[e][r][l][a]+d[t][n][a];i>o&&(o=i,s=t,c=n)}}}ee[s][c][a]+=D[e][t][r][a]}}}return{dataId:l.write(ci(ee,r.dtype),i.shape,i.dtype),shape:i.shape,dtype:i.dtype}}},yA={kernelName:Et,backendName:`cpu`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,filter:i,dy:a}=e,{strides:o,pad:s,dilations:c}=n,l=t,u=_e(r.shape,l.data.get(r.dataId).values),d=_e(i.shape,l.data.get(i.dataId).values),{batchSize:f,inHeight:p,inWidth:m,inChannels:h,outHeight:g,outWidth:_,padInfo:v,strideHeight:y,strideWidth:x,filterHeight:S,filterWidth:C,dilationHeight:w,dilationWidth:T,outShape:E}=$o(r.shape,i.shape,o,s,`NHWC`,c);b(a.rank===E.length,()=>`Error in ${Et}, dy must have the same rank as output ${E.length}, but got ${a.rank}`);let D=_e(E,l.data.get(a.dataId).values),ee=xe(r.shape,r.dtype);for(let e=0;e<f;++e)for(let t=0;t<g;++t){let n=t*y-v.top;for(let r=0;r<_;++r){let i=r*x-v.left;for(let a=0;a<h;++a){let o=-(2**53-1),s=n<0?0:n,c=i<0?0:i;for(let t=0;t<S;++t){let r=n+t*w;if(r>=0&&r<p)for(let n=0;n<C;++n){let l=i+n*T;if(l>=0&&l<m){let i=u[e][r][l][a]+d[t][n][a];i>o&&(o=i,s=r,c=l)}}}ee[e][s][c][a]+=D[e][t][r][a]}}}return{dataId:l.write(ci(ee,r.dtype),r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}};function bA(e){let{inputs:t,backend:n,attrs:r}=e,{image:i}=t,{canvas:a,options:o}=r,{contextOptions:s,imageOptions:c}=o||{},l=c?.alpha||1,u=s?.contextType||`2d`;if(u!==`2d`)throw Error(`Context type ${s.contextType} is not supported by the CPU backend.`);let d=a.getContext(u,s?.contextAttributes||{});if(d==null)throw Error(`Could not get the context with ${u} type.`);let[f,p]=i.shape.slice(0,2),m=i.shape.length===2?1:i.shape[2],h=n.data.get(i.dataId).values,g=i.dtype===`float32`?255:1,_=new Uint8ClampedArray(p*f*4);for(let e=0;e<f*p;++e){let t=[0,0,0,255*l];for(let n=0;n<m;n++){let r=h[e*m+n];if(i.dtype===`float32`){if(r<0||r>1)throw Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${r}.`)}else if(i.dtype===`int32`&&(r<0||r>255))throw Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${r}.`);m===1?(t[0]=r*g,t[1]=r*g,t[2]=r*g):t[n]=r*g}let n=e*4;_[n+0]=Math.round(t[0]),_[n+1]=Math.round(t[1]),_[n+2]=Math.round(t[2]),_[n+3]=Math.round(t[3])}a.width=p,a.height=f;let v=new ImageData(_,p,f);return d.putImageData(v,0,0),i}var xA={kernelName:Ot,backendName:`cpu`,kernelFunc:bA};function SA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;J(i,`sum`);let s;s=i.dtype===`bool`?AE({inputs:{x:i},backend:n,attrs:{dtype:`int32`}}):TE({inputs:{x:i},backend:n});let c=s.shape.length,l=O(a,s.shape),u=ol(l,c),d=l,f=s;u!=null&&(f=MD({inputs:{x:s},backend:n,attrs:{perm:u}}),d=cl(d.length,c)),al(`sum`,d,f.shape.length);let[p,m]=rl(f.shape,d),h=wE(n,p,Vi(f.dtype,`int32`)),g=C(m),_=n.data.get(h.dataId).values,v=n.data.get(f.dataId).values;for(let e=0;e<_.length;++e){let t=e*g,n=0;for(let e=0;e<g;++e)n+=v[t+e];_[e]=n}if(o){let e=il(h.shape,l),t=h;h=WO({inputs:{x:h},backend:n,attrs:{shape:e}}),n.disposeIntermediateTensorInfo(t)}return n.disposeIntermediateTensorInfo(s),u!=null&&n.disposeIntermediateTensorInfo(f),h}var CA={kernelName:`Sum`,backendName:`cpu`,kernelFunc:SA};function wA(e){let{inputs:t,backend:n,attrs:r}=e,{equation:i}=r,a=t,{allDims:o,summedDims:s,idDims:c}=Qh(i,a.length);eg(o.length,c,a);let{path:l,steps:u}=tg(s,c),d=u.length,f=null,p=o.length,m=[];for(let e=0;e<d;++e){for(let t of u[e]){let{permutationIndices:e,expandDims:r}=$h(p,c[t]),i;ng(e)?i=a[t]:(i=MD({inputs:{x:a[t]},backend:n,attrs:{perm:e}}),m.push(i));let o=i.shape.slice();for(let e=0;e<r.length;++e)o.splice(r[e],0,1);w(i.shape,o)||(i=WO({inputs:{x:i},backend:n,attrs:{shape:o}}),m.push(i)),f===null?f=i:(f=wD({inputs:{a:i,b:f},backend:n}),m.push(f))}e<d-1&&(l[e]>=0&&(f=SA({inputs:{x:f},backend:n,attrs:{axis:l[e]-(o.length-p),keepDims:!1}}),m.push(f)),p--)}for(let e of m)e!==f&&n.disposeIntermediateTensorInfo(e);return f}var TA={kernelName:At,backendName:`cpu`,kernelFunc:wA};function EA(e){let{inputs:t,backend:n}=e,{dy:r,y:i}=t;J([r,i],`eluGrad`);let a=new Float32Array(C(i.shape)),o=n.data.get(i.dataId).values,s=n.data.get(r.dataId).values;for(let e=0;e<o.length;++e){let t=o[e];t>=0?a[e]=s[e]:a[e]=s[e]*(t+1)}return n.makeTensorInfo(i.shape,`float32`,a)}var DA={kernelName:jt,backendName:`cpu`,kernelFunc:EA},OA=Ph,kA=Fh,AA=Ih,jA=Lh,MA=Rh,NA=zh,PA={kernelName:`Erf`,backendName:`cpu`,kernelFunc:HE(`Erf`,e=>{let t=Math.sign(e),n=Math.abs(e),r=1/(1+OA*n);return t*(1-((((NA*r+MA)*r+jA)*r+AA)*r+kA)*r*Math.exp(-n*n))})};function FA(e){let{inputs:t,backend:n,attrs:r}=e,{input:i}=t,{dim:a}=r,o=i.shape.length,s=i.shape.slice(),c=a;return a<0&&(b(-(o+1)<=a,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+a+1),s.splice(c,0,1),WO({inputs:{x:i},backend:n,attrs:{shape:s}})}var IA={kernelName:Nt,backendName:`cpu`,kernelFunc:FA},LA=ME(kt,xE((e,t)=>e/t)),RA={kernelName:kt,backendName:`cpu`,kernelFunc:LA};function zA(e,t,n){let r=e.shape,i=r[0],a=r[1],o=n.data.get(e.dataId),s=o.complexTensorInfos.real,c=o.complexTensorInfos.imag,l=[i,a],u=C(l),d=re(`float32`,u),f=re(`float32`,u);for(let e=0;e<i;e++){let r=oO({inputs:{x:s},backend:n,attrs:{begin:[e,0],size:[1,a]}}),i=oO({inputs:{x:c},backend:n,attrs:{begin:[e,0],size:[1,a]}}),o=SE({inputs:{real:r,imag:i},backend:n}),{real:l,imag:u}=BA(o,t,n),p=Bh(l,u);for(let t=0;t<a;t++){let n=Wh(p,t);d[e*a+t]=n.real,f[e*a+t]=n.imag}n.disposeIntermediateTensorInfo(r),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(o)}let p=n.makeTensorInfo(l,`float32`,d),m=n.makeTensorInfo(l,`float32`,f),h=SE({inputs:{real:p,imag:m},backend:n});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),h}function BA(e,t,n){let r=C(e.shape),i=n.data.get(e.dataId),a=n.data.get(i.complexTensorInfos.real.dataId).values,o=n.data.get(i.complexTensorInfos.imag.dataId).values;if(VA(r)){let i=HA(a,o,r,t,n),s=[e.shape[0],e.shape[1]];if(t){let e=n.makeTensorInfo(s,`float32`,i.real),t=n.makeTensorInfo(s,`float32`,i.imag),a=n.makeTensorInfo([],`float32`,oi(r,`float32`)),o=TE({inputs:{x:a},backend:n}),c=RA.kernelFunc({inputs:{a:e,b:a},backend:n}),l=RA.kernelFunc({inputs:{a:t,b:o},backend:n}),u=n.data.get(c.dataId).values,d=n.data.get(l.dataId).values;return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(a),n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(c),n.disposeIntermediateTensorInfo(l),{real:u,imag:d}}return i}return Vh(UA(Bh(a,o),r,t))}function VA(e){return!(e&e-1)}function HA(e,t,n,r,i){if(n===1)return{real:e,imag:t};let a=Bh(e,t),o=n/2,s=Hh(a),c=s.real,l=s.imag,u=[c.length],d=i.makeTensorInfo(u,`float32`,c),f=i.makeTensorInfo(u,`float32`,l),p=SE({inputs:{real:d,imag:f},backend:i}),m=Uh(a),h=m.real,g=m.imag,_=[h.length],v=i.makeTensorInfo(_,`float32`,h),y=i.makeTensorInfo(_,`float32`,g),b=SE({inputs:{real:v,imag:y},backend:i}),x=HA(c,l,o,r,i),S=x.real,C=x.imag,w=[S.length],T=i.makeTensorInfo(w,`float32`,S),E=i.makeTensorInfo(w,`float32`,C),D=SE({inputs:{real:T,imag:E},backend:i}),ee=HA(h,g,o,r,i),te=ee.real,O=ee.imag,ne=[te.length],re=i.makeTensorInfo(ne,`float32`,te),ie=i.makeTensorInfo(ne,`float32`,O),ae=SE({inputs:{real:re,imag:ie},backend:i}),oe=Kh(n,r),se=[oe.real.length],ce=i.makeTensorInfo(se,`float32`,oe.real),le=i.makeTensorInfo(se,`float32`,oe.imag),ue=SE({inputs:{real:ce,imag:le},backend:i}),de=wD({inputs:{a:ue,b:ae},backend:i}),fe=FE({inputs:{a:D,b:de},backend:i}),pe=wO({inputs:{a:D,b:de},backend:i}),me=DE({inputs:{input:fe},backend:i}),he=DE({inputs:{input:pe},backend:i}),k=Fk({inputs:{input:fe},backend:i}),ge=Fk({inputs:{input:pe},backend:i}),_e=Lk({inputs:[me,he],backend:i,attrs:{axis:0}}),ve=Lk({inputs:[k,ge],backend:i,attrs:{axis:0}}),ye=i.data.get(_e.dataId).values,be=i.data.get(ve.dataId).values;return i.disposeIntermediateTensorInfo(d),i.disposeIntermediateTensorInfo(f),i.disposeIntermediateTensorInfo(p),i.disposeIntermediateTensorInfo(v),i.disposeIntermediateTensorInfo(y),i.disposeIntermediateTensorInfo(b),i.disposeIntermediateTensorInfo(T),i.disposeIntermediateTensorInfo(E),i.disposeIntermediateTensorInfo(D),i.disposeIntermediateTensorInfo(re),i.disposeIntermediateTensorInfo(ie),i.disposeIntermediateTensorInfo(ae),i.disposeIntermediateTensorInfo(ce),i.disposeIntermediateTensorInfo(le),i.disposeIntermediateTensorInfo(ue),i.disposeIntermediateTensorInfo(de),i.disposeIntermediateTensorInfo(fe),i.disposeIntermediateTensorInfo(pe),i.disposeIntermediateTensorInfo(me),i.disposeIntermediateTensorInfo(k),i.disposeIntermediateTensorInfo(he),i.disposeIntermediateTensorInfo(ge),i.disposeIntermediateTensorInfo(_e),i.disposeIntermediateTensorInfo(ve),{real:ye,imag:be}}function UA(e,t,n){let r=new Float32Array(t*2);for(let i=0;i<t;i++){let a=0,o=0;for(let r=0;r<t;r++){let s=qh(i*r,t,n),c=Wh(e,r);a+=c.real*s.real-c.imag*s.imag,o+=c.real*s.imag+c.imag*s.real}n&&(a/=t,o/=t),Gh(r,a,o,i)}return r}function WA(e){let{inputs:t,backend:n}=e,{input:r}=t,i=C(r.shape),a=r.shape[r.shape.length-1],o=i/a,s=WO({inputs:{x:r},backend:n,attrs:{shape:[o,a]}}),c=zA(s,!1,n),l=WO({inputs:{x:c},backend:n,attrs:{shape:r.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(c),l}var GA={kernelName:`FFT`,backendName:`cpu`,kernelFunc:WA};function KA(e){let{backend:t,attrs:n}=e,{shape:r,value:i,dtype:a}=n,o=a||pe(i),s=ie(o,C(r));return JA(s,i,o),t.makeTensorInfo(r,o,s)}var qA={kernelName:Ft,backendName:`cpu`,kernelFunc:KA};function JA(e,t,n){e.fill(t)}var YA={kernelName:It,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,i=n,a=re(r.dtype,C(r.shape)),[o,s,c,l]=r.shape,u=i.data.get(r.dataId).values;for(let e=0;e<o;e++){let t=e*c*s*l;for(let e=0;e<s;e++){let n=c*l*e;for(let e=0;e<c;e++){let r=e*l;for(let i=0;i<l;i++){let o=Math.round(c-e-1),s=t+n+r+i,d=u[s];if(o>=0&&o<c){let e=o*l;d=u[t+n+e+i]}a[s]=d}}}}return{dataId:i.write(a,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}};function XA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=zk({inputs:{x:i,filter:a},backend:n,attrs:{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f}});if(o){let e=h;if(u===`NCHW`&&o.shape.length===1&&o.shape[0]!==1){let e=WO({inputs:{x:o},backend:n,attrs:{shape:[o.shape[0],1,1]}});h=FE({inputs:{a:h,b:e},backend:n}),n.disposeIntermediateTensorInfo(e)}else h=FE({inputs:{a:h,b:o},backend:n});n.disposeIntermediateTensorInfo(e)}if(p){let e=h;if(u===`NCHW`&&p===`prelu`&&s.shape.length===1&&s.shape[0]!==1){let e=WO({inputs:{x:s},backend:n,attrs:{shape:[s.shape[0],1,1]}});h=UO(n,h,p,e,m),n.disposeIntermediateTensorInfo(e)}else h=UO(n,h,p,s,m);n.disposeIntermediateTensorInfo(e)}return h}var ZA={kernelName:kr,backendName:`cpu`,kernelFunc:XA};function QA(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=lA({inputs:{x:i,filter:a},backend:n,attrs:{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f}});if(o){let e=h;h=FE({inputs:{a:h,b:o},backend:n}),n.disposeIntermediateTensorInfo(e)}if(p){let e=h;h=UO(n,h,p,s,m),n.disposeIntermediateTensorInfo(e)}return h}var $A={kernelName:Ar,backendName:`cpu`,kernelFunc:QA};function ej(e){let{inputs:t,backend:n}=e,{params:r,indices:i}=t,a=C(r.shape),o=i.shape,s=o[o.length-1],[c,l,u,d]=qm(r,i);if(l===0)return n.makeTensorInfo(c,r.dtype,[]);let f=n.data.get(i.dataId).values,p=aD(f,n.bufferSync(r),r.dtype,l,s,u,d,r.shape,a);return n.makeTensorInfo(c,r.dtype,p.values)}var tj={kernelName:Vt,backendName:`cpu`,kernelFunc:ej};function nj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,indices:a}=t,{axis:o,batchDims:s}=r;J([i,a],`gatherV2`);let c=O(o,i.shape)[0],l=n.data.get(a.dataId).values,u=i.shape[c];for(let e=0;e<l.length;++e){let t=l[e];b(t<=u-1&&t>=0,()=>`GatherV2: the index value ${t} is not in [0, ${u-1}]`)}let d=s;s??(d=0);let f=C(a.shape),p=bg(i,a,c,d),m=WO({inputs:{x:i},backend:n,attrs:{shape:[p.batchSize,p.outerSize,p.dimSize,p.sliceSize]}}),h=WO({inputs:{x:a},backend:n,attrs:{shape:[p.batchSize,f/p.batchSize]}}),g=[p.batchSize,p.outerSize,f/p.batchSize,p.sliceSize],_=n.bufferSync(h),v=oD(n.bufferSync(m),_,g);return n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),n.makeTensorInfo(p.outputShape,v.dtype,v.values)}var rj={kernelName:Bt,backendName:`cpu`,kernelFunc:nj};function ij(e){let{inputs:t,backend:n}=e,{input:r}=t,i=C(r.shape),a=r.shape[r.shape.length-1],o=i/a,s=WO({inputs:{x:r},backend:n,attrs:{shape:[o,a]}}),c=zA(s,!0,n),l=WO({inputs:{x:c},backend:n,attrs:{shape:r.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(c),l}var aj={kernelName:Gt,backendName:`cpu`,kernelFunc:ij},oj={kernelName:qt,backendName:`cpu`,kernelFunc:HE(qt,e=>+!!Number.isFinite(e),`bool`)},sj={kernelName:Jt,backendName:`cpu`,kernelFunc:HE(Jt,e=>+(Math.abs(e)===1/0),`bool`)},cj={kernelName:Yt,backendName:`cpu`,kernelFunc:HE(Yt,e=>+!!Number.isNaN(e),`bool`)};function lj(e){let{backend:t,attrs:n}=e,{start:r,stop:i,num:a}=n,o=hD(r,i,a);return t.makeTensorInfo([o.length],`float32`,o)}var uj={kernelName:$t,backendName:`cpu`,kernelFunc:lj},dj={kernelName:en,backendName:`cpu`,kernelFunc:HE(en,e=>Math.log1p(e))},fj={kernelName:tn,backendName:`cpu`,kernelFunc:ME(tn,xE((e,t)=>e&&t),null,`bool`)},pj={kernelName:nn,backendName:`cpu`,kernelFunc:HE(nn,e=>+!e,`bool`)},mj={kernelName:rn,backendName:`cpu`,kernelFunc:ME(rn,xE((e,t)=>e||t),null,`bool`)};function hj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r;J(i,`LRN`);let l=i.shape[3],u=l-1,d=n.data.get(i.dataId).values,f=C(i.shape),p=new Float32Array(f);function m(e){let t=e%l,n=e-t+Math.max(0,t-a),r=e-t+Math.min(t+a,u),i=0;for(;n<=r;n++){let e=d[n];i+=e*e}return i}for(let e=0;e<f;e++){let t=m(e),n=d[e]*(o+s*t)**+-c;p[e]=n}return n.makeTensorInfo(i.shape,i.dtype,p)}var gj={kernelName:`LRN`,backendName:`cpu`,kernelFunc:hj};function _j(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r;J(o,`LRNGrad`);let d=C(o.shape),f=o.shape[3],p=n.data.get(o.dataId).values,m=n.data.get(i.dataId).values,h=n.data.get(a.dataId).values,g=new Float32Array(d),_=d;for(let e=0;e<_;e++){let t=e%f,n=e-t+Math.max(0,t-s),r=e-t+Math.min(f,t+s+1),i=0;for(let e=n;e<r;e++)i+=m[e]**2;i=l*i+c;for(let t=n;t<r;t++){let n=-2*l*u*m[t]*h[e]/i;e===t&&(n+=i**+-u),n*=p[e],g[t]+=n}}return n.makeTensorInfo(o.shape,i.dtype,g)}var vj={kernelName:on,backendName:`cpu`,kernelFunc:_j};function yj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reductionIndices:a,keepDims:o}=r,s=n,c=i.shape,l=c.length,u=O(a,c),d=u,f=ol(d,l),p=s.data.get(i.dataId).values;if(f!=null){let e=Array(l);for(let t=0;t<e.length;t++)e[t]=c[f[t]];p=jD(p,c,i.dtype,f,e),d=cl(d.length,l),c=e}J(i,`max`),al(`max`,d,l);let[m,h]=rl(c,d),g=C(h),_=vD(p,g,m,i.dtype),v=s.write(_,m,i.dtype),y=m;return o&&(y=il(m,u)),{dataId:v,shape:y,dtype:i.dtype}}var bj={kernelName:`Max`,backendName:`cpu`,kernelFunc:yj};function xj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;J(i,`maxPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;b(ms(o,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=es(i.shape,a,o,1,s,c),u;if(l.filterWidth===1&&l.filterHeight===1&&w(l.inShape,l.outShape))u=TE({inputs:{x:i},backend:n});else{let e=n.data.get(i.dataId).values,t=k(i.shape),r=pk(e,i.shape,i.dtype,t,l,`max`);u=n.makeTensorInfo(l.outShape,i.dtype,r.values)}return u}var Sj={kernelName:cn,backendName:`cpu`,kernelFunc:xj};function Cj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r;J(i,`maxPool3d`);let u=ts(i.shape,a,o,1,s,c,l),d=n.data.get(i.dataId).values,f=hk(d,i.shape,i.dtype,k(i.shape),u,`max`);return n.makeTensorInfo(f.shape,`float32`,f.values)}var wj={kernelName:un,backendName:`cpu`,kernelFunc:Cj};function Tj(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,{filterSize:o,strides:s,pad:c,dimRoundingMode:l}=r;J([i,a],`maxPool3DGrad`);let u=ts(a.shape,o,s,1,c,l),d=gk(n.bufferSync(a),u),f=u.strideDepth,p=u.strideHeight,m=u.strideWidth,h=u.dilationDepth,g=u.dilationHeight,_=u.dilationWidth,v=u.effectiveFilterDepth,y=u.effectiveFilterHeight,b=u.effectiveFilterWidth,x=v-1-u.padInfo.front,S=b-1-u.padInfo.left,C=y-1-u.padInfo.top,w=_o(a.shape,`float32`),T=n.bufferSync(i);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let n=0;n<u.inDepth;++n)for(let r=0;r<u.inHeight;++r)for(let i=0;i<u.inWidth;++i){let a=n-x,o=r-C,s=i-S,c=0;for(let n=0;n<v;n+=h){let r=(a+n)/f;if(!(r<0||r>=u.outDepth||Math.floor(r)!==r))for(let i=0;i<y;i+=g){let a=(o+i)/p;if(!(a<0||a>=u.outHeight||Math.floor(a)!==a))for(let o=0;o<b;o+=_){let l=(s+o)/m;if(l<0||l>=u.outWidth||Math.floor(l)!==l)continue;let f=+(v*y*b-1-d.get(e,r,a,l,t)===n*y*b+i*b+o);if(f===0)continue;let p=T.get(e,r,a,l,t);c+=p*f}}}w.set(c,e,n,r,i,t)}return n.makeTensorInfo(w.shape,w.dtype,w.values)}var Ej={kernelName:dn,backendName:`cpu`,kernelFunc:Tj};function Dj(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a,output:o}=t,s=a;J([a,o],`maxPoolGrad`);let{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=r,f=es(s.shape,c,l,1,u,d),p=n.data.get(s.dataId).values,m=_o(f.outShape,s.dtype,mk(p,s.shape,s.dtype,f).values),h=f.strideHeight,g=f.strideWidth,_=f.dilationHeight,v=f.dilationWidth,y=f.effectiveFilterHeight,b=f.effectiveFilterWidth,x=b-1-f.padInfo.left,S=y-1-f.padInfo.top,C=_o(s.shape,`float32`),w=n.data.get(i.dataId).values,T=_o(i.shape,`float32`,w);for(let e=0;e<f.batchSize;++e)for(let t=0;t<f.inChannels;++t)for(let n=0;n<f.inHeight;++n)for(let r=0;r<f.inWidth;++r){let i=n-S,a=r-x,o=0;for(let n=0;n<y;n+=_){let r=(i+n)/h;if(!(r<0||r>=f.outHeight||Math.floor(r)!==r))for(let i=0;i<b;i+=v){let s=(a+i)/g;if(s<0||s>=f.outWidth||Math.floor(s)!==s)continue;let c=+(y*b-1-m.get(e,r,s,t)===n*b+i);if(c===0)continue;let l=T.get(e,r,s,t);o+=l*c}}C.set(o,e,n,r,t)}return n.makeTensorInfo(C.shape,C.dtype,C.values)}var Oj={kernelName:ln,backendName:`cpu`,kernelFunc:Dj};function kj(e,t,n,r,i){let a=pk(e,t,n,k(t),i,`max`),o=mk(e,t,n,i,!0,r);return[a.values,o.values]}var Aj={kernelName:fn,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{filterSize:i,strides:a,pad:o,includeBatchInIndex:s}=t,c=n;J(r,`MaxPoolWithArgmax`);let l=c.data.get(r.dataId).values,u=es(r.shape,i,a,[1,1],o),[d,f]=kj(l,r.shape,r.dtype,s,u),p=c.write(d,u.outShape,r.dtype),m=c.write(f,u.outShape,r.dtype);return[{dataId:p,shape:u.outShape,dtype:r.dtype},{dataId:m,shape:u.outShape,dtype:`int32`}]}};function jj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=O(a,i.shape),c=rl(i.shape,s)[1],l=C(c),u=[],d=n.makeTensorInfo([],`float32`,new Float32Array([l]));u.push(d);let f=AE({inputs:{x:i},backend:n,attrs:{dtype:`float32`}});u.push(f);let p=LA({inputs:{a:f,b:d},backend:n});u.push(p);let m=SA({inputs:{x:p},backend:n,attrs:{axis:a,keepDims:o}});return u.forEach(e=>n.disposeIntermediateTensorInfo(e)),m}var Mj={kernelName:pn,backendName:`cpu`,kernelFunc:jj};function Nj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;J(i,`min`);let s=O(a,i.shape),c=s,l=ol(c,i.shape.length),u=i;l!=null&&(u=MD({inputs:{x:i},backend:n,attrs:{perm:l}}),c=cl(c.length,i.shape.length)),al(`min`,c,u.shape.length);let[d,f]=rl(u.shape,c),p=C(f),m=be(C(d),u.dtype),h=n.data.get(u.dataId).values;for(let e=0;e<m.length;++e){let t=e*p,n=h[t];for(let e=0;e<p;++e){let r=h[t+e];(Number.isNaN(r)||r<n)&&(n=r)}m[e]=n}l!=null&&n.disposeIntermediateTensorInfo(u);let g=n.makeTensorInfo(d,u.dtype,m);if(o){let e=il(d,s),t=WO({inputs:{x:g},backend:n,attrs:{shape:e}});return n.disposeIntermediateTensorInfo(g),t}return g}var Pj={kernelName:`Min`,backendName:`cpu`,kernelFunc:Nj};function Fj(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,mode:o}=r;J(i,`mirrorPad`);let s=a.map((e,t)=>e[0]+i.shape[t]+e[1]),c=a.map(e=>e[0]),l=a.map((e,t)=>e[0]+i.shape[t]),u=o===`reflect`?0:1,d=n.data.get(i.dataId).values,f=i.shape.length,p=k(i.shape),m=C(s),h=s.length,g=k(s),_=re(i.dtype,m);for(let e=0;e<m;e++){let t=we(e,h,g);for(let e=0;e<h;e++)t[e]<c[e]?t[e]=c[e]*2-t[e]-u:t[e]>=l[e]&&(t[e]=(l[e]-1)*2-t[e]+u);t=t.map((e,t)=>e-c[t]);let n=Ce(t,f,p);_[e]=d[n]}return{dataId:n.write(_,s,i.dtype),shape:s,dtype:i.dtype}}var Ij={kernelName:hn,backendName:`cpu`,kernelFunc:Fj},Lj={kernelName:`Mod`,backendName:`cpu`,kernelFunc:ME(`Mod`,xE(((e,t)=>{let n=e%t;return e<0&&t<0||e>=0&&t>=0?n:(n+t)%t})))};function Rj(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{dim:a}=r,o=i.shape.length,s=a;if(s===-1&&(s=o-1),s!==o-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${o} and dim was ${s}`);let c=O([s],i.shape),l=yj({inputs:{x:i},backend:n,attrs:{reductionIndices:c,keepDims:!1}}),u=il(l.shape,c),d=WO({inputs:{x:l},backend:n,attrs:{shape:u}}),f=wO({inputs:{a:i,b:d},backend:n}),p=ZE({inputs:{x:f},backend:n}),m=SA({inputs:{x:p},backend:n,attrs:{axis:c,keepDims:!1}}),h=WO({inputs:{x:m},backend:n,attrs:{shape:u}}),g=LA({inputs:{a:p,b:h},backend:n});return n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),g}var zj={kernelName:rr,backendName:`cpu`,kernelFunc:Rj};function Bj(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r;J(i,`multinomial`);let c=s?i:Rj({inputs:{logits:i},backend:n,attrs:{dim:-1}}),l=c.shape[0],u=c.shape[1],d=n.data.get(c.dataId).values,f=[l,a],p=be(C(f),`int32`);for(let e=0;e<l;++e){let t=e*u,n=new Float32Array(u-1);n[0]=d[t];for(let e=1;e<n.length;++e)n[e]=n[e-1]+d[t+e];let r=yd.alea(o.toString()),i=e*a;for(let e=0;e<a;++e){let t=r();p[i+e]=n.length;for(let r=0;r<n.length;r++)if(t<n[r]){p[i+e]=r;break}}}return s||n.disposeIntermediateTensorInfo(c),n.makeTensorInfo(f,`int32`,p)}var Vj={kernelName:gn,backendName:`cpu`,kernelFunc:Bj},Hj=Ap;function Uj(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=r;J(i,`NonMaxSuppression`);let l=n.data.get(i.dataId).values,u=n.data.get(a.dataId).values,{selectedIndices:d}=Hj(l,u,o,s,c);return n.makeTensorInfo([d.length],`int32`,new Int32Array(d))}var Wj={kernelName:yn,backendName:`cpu`,kernelFunc:Uj},Gj=jp;function Kj(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,padToMaxOutputSize:l}=r;J(i,`NonMaxSuppressionPadded`);let u=n.data.get(i.dataId).values,d=n.data.get(a.dataId).values,{selectedIndices:f,validOutputs:p}=Gj(u,d,o,s,c,l);return[n.makeTensorInfo([f.length],`int32`,new Int32Array(f)),n.makeTensorInfo([],`int32`,new Int32Array([p]))]}var qj={kernelName:bn,backendName:`cpu`,kernelFunc:Kj},Jj=Mp;function Yj(e){let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=r;J(i,`NonMaxSuppressionWithScore`);let u=n.data.get(i.dataId).values,d=n.data.get(a.dataId).values,{selectedIndices:f,selectedScores:p}=Jj(u,d,o,s,c,l);return[n.makeTensorInfo([f.length],`int32`,new Int32Array(f)),n.makeTensorInfo([p.length],`float32`,new Float32Array(p))]}var Xj={kernelName:xn,backendName:`cpu`,kernelFunc:Yj};function Zj(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r;J(i,`oneHot`);let l=C(i.shape),u=new Float32Array(l*o);u.fill(c);let d=n.data.get(i.dataId).values;for(let e=0;e<l;++e)d[e]>=0&&d[e]<o&&(u[e*o+d[e]]=s);return n.makeTensorInfo([...i.shape,o],a,u)}var Qj={kernelName:Cn,backendName:`cpu`,kernelFunc:Zj};function $j(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`zerosLike is not supported for string tensors`);if(r.dtype===`complex64`){let e=DE({inputs:{input:r},backend:n}),t=$j({inputs:{x:e},backend:n}),i=Fk({inputs:{input:r},backend:n}),a=$j({inputs:{x:i},backend:n}),o=SE({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return KA({backend:n,attrs:{shape:r.shape,value:0,dtype:r.dtype}})}var eM={kernelName:wr,backendName:`cpu`,kernelFunc:$j};function tM(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`onesLike is not supported for string tensors`);if(r.dtype===`complex64`){let e=DE({inputs:{input:r},backend:n}),t=tM({inputs:{x:e},backend:n}),i=Fk({inputs:{input:r},backend:n}),a=$j({inputs:{x:i},backend:n}),o=SE({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return KA({backend:n,attrs:{shape:r.shape,value:1,dtype:r.dtype}})}var nM={kernelName:Sn,backendName:`cpu`,kernelFunc:tM};function rM(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return FA({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{x(a,e.shape,`All tensors passed to stack must have matching shapes`),b(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=Lk({inputs:t.map(e=>{let t=FA({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeIntermediateTensorInfo(e)),c}var iM={kernelName:wn,backendName:`cpu`,kernelFunc:rM};function aM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,constantValue:o}=r;J(i,`pad`);let s=a.map((e,t)=>e[0]+i.shape[t]+e[1]),c=a.map(e=>e[0]),l=n.data.get(i.dataId).values,u=C(i.shape),d=i.shape.length,f=k(i.shape),p=C(s),m=s.length,h=k(s),g=re(i.dtype,p);o!==0&&g.fill(o);for(let e=0;e<u;e++){let t=Ce(we(e,d,f).map((e,t)=>e+c[t]),m,h);g[t]=l[e]}return{dataId:n.write(g,s,i.dtype),shape:s,dtype:i.dtype}}var oM={kernelName:Tn,backendName:`cpu`,kernelFunc:aM},sM={kernelName:`Pow`,backendName:`cpu`,kernelFunc:ME(`Pow`,xE((e,t)=>e**+t))};function cM(e){let{inputs:t,backend:n,attrs:r}=e,{paramsNestedSplits:i,paramsDenseValues:a,indices:o}=t,{outputRaggedRank:s}=r,c=i.map(e=>n.data.get(e.dataId).values),l=i.map(e=>e.shape),u=n.data.get(a.dataId).values,d=n.data.get(o.dataId).values,[f,p,m]=WD(c,l,u,a.shape,a.dtype,d,o.shape,s),h=f.map(e=>n.makeTensorInfo([e.length],`int32`,e)),g=n.makeTensorInfo(m,a.dtype,p);return h.concat([g])}var lM={kernelName:On,backendName:`cpu`,kernelFunc:cM};function uM(e){let{inputs:t,backend:n}=e,{starts:r,limits:i,deltas:a}=t,o=n.data.get(r.dataId).values,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,[l,u]=KD(o,r.shape,r.dtype,s,i.shape,c,a.shape);return[n.makeTensorInfo([l.length],`int32`,l),n.makeTensorInfo([u.length],r.dtype,u)]}var dM={kernelName:kn,backendName:`cpu`,kernelFunc:uM};function fM(e){let{inputs:t,backend:n,attrs:r}=e,{shape:i,values:a,defaultValue:o,rowPartitionTensors:s}=t,{rowPartitionTypes:c}=r,l=n.data.get(i.dataId).values,u=n.data.get(a.dataId).values,d=n.data.get(o.dataId).values,f=s.map(e=>n.data.get(e.dataId).values),p=s.map(e=>e.shape),[m,h]=ZD(l,i.shape,u,a.shape,a.dtype,d,o.shape,f,p,c);return n.makeTensorInfo(m,a.dtype,h)}var pM={kernelName:An,backendName:`cpu`,kernelFunc:fM};function mM(e){let{backend:t,attrs:n}=e,{start:r,stop:i,dtype:a,step:o}=n,s=QD(r,i,o,a);return t.makeTensorInfo([s.length],a,s)}var hM={kernelName:jn,backendName:`cpu`,kernelFunc:mM},gM={kernelName:Nn,backendName:`cpu`,kernelFunc:HE(Nn,e=>1/e)};function _M(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r;J(i,`resizeBilinear`);let c=k(i.shape),[l,u]=s,[d,f,p,m]=i.shape,h=n.data.get(i.dataId).values,g=new Float32Array(C([d,l,u,m])),_=[a&&l>1?f-1:f,a&&u>1?p-1:p],v=[a&&l>1?l-1:l,a&&u>1?u-1:u],y=0,b=_[0]/v[0],x=_[1]/v[1];for(let e=0;e<d;e++)for(let t=0;t<l;t++){let n;n=o?b*(t+.5)-.5:b*t;let r=Math.max(0,Math.floor(n)),i=n-r,a=Math.min(f-1,Math.ceil(n)),s=e*c[0]+r*c[1],l=e*c[0]+a*c[1];for(let e=0;e<u;e++){let t;t=o?x*(e+.5)-.5:x*e;let n=Math.max(0,Math.floor(t)),r=t-n,a=Math.min(p-1,Math.ceil(t)),u=s+n*c[2],d=l+n*c[2],f=s+a*c[2],_=l+a*c[2];for(let e=0;e<m;e++){let t=h[u+e],n=h[d+e],a=h[f+e],o=h[_+e],s=t+(a-t)*r,c=s+(n+(o-n)*r-s)*i;g[y++]=c}}}return n.makeTensorInfo([d,l,u,m],`float32`,g)}var vM={kernelName:Rn,backendName:`cpu`,kernelFunc:_M};function yM(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r;J([a,i],`resizeBilinearGrad`);let s=k(i.shape),[c,l,u,d]=i.shape,[,f,p]=a.shape,m=new Float32Array(c*l*u*d),h=[o&&f>1?l-1:l,o&&p>1?u-1:u],g=[o&&f>1?f-1:f,o&&p>1?p-1:p],_=h[0]/g[0],v=h[1]/g[1],y=n.data.get(a.dataId).values,b=0;for(let e=0;e<c;e++){let t=e*s[0];for(let e=0;e<f;e++){let n=e*_,r=Math.floor(n),i=Math.min(Math.ceil(n),l-1),a=t+r*s[1],o=t+i*s[1],c=n-r,f=1-c;for(let e=0;e<p;e++){let t=e*v,n=Math.floor(t),r=Math.min(Math.ceil(t),u-1),i=t-n,l=1-i,p=a+n*s[2],h=a+r*s[2],g=o+n*s[2],_=o+r*s[2],x=f*l,S=f*i,C=c*l,w=c*i;for(let e=0;e<d;e++){let t=y[b++];m[p+e]+=t*x,m[h+e]+=t*S,m[g+e]+=t*C,m[_+e]+=t*w}}}}return n.makeTensorInfo([c,u,l,d],`float32`,m)}var bM={kernelName:zn,backendName:`cpu`,kernelFunc:yM};function xM(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r;J(i,`resizeNearestNeighbor`);let c=k(i.shape),[l,u]=s,[d,f,p,m]=i.shape,h=n.data.get(i.dataId).values,g=new Float32Array(d*l*u*m),_=[a&&l>1?f-1:f,a&&u>1?p-1:p],v=[a&&l>1?l-1:l,a&&u>1?u-1:u],y=_[0]/v[0],b=_[1]/v[1],x=0;for(let e=0;e<d;e++){let t=e*c[0];for(let e=0;e<l;e++){let n=o?y*(e+.5):y*e,r=Math.min(f-1,a?Math.round(n):Math.floor(n));o&&(r=Math.max(0,r));let i=t+r*c[1];for(let e=0;e<u;e++){let t=o?b*(e+.5):b*e,n=Math.min(p-1,a?Math.round(t):Math.floor(t));o&&(n=Math.max(0,n));let r=i+n*c[2];for(let e=0;e<m;e++){let t=h[r+e];g[x++]=t}}}}return n.makeTensorInfo([d,l,u,m],i.dtype,g)}var SM={kernelName:In,backendName:`cpu`,kernelFunc:xM};function CM(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r;J([a,i],`resizeNearestNeighborGrad`);let s=k(i.shape),c=k(a.shape),[l,u,d,f]=i.shape,[,p,m]=a.shape,h=new Float32Array(l*u*d*f),g=n.data.get(a.dataId).values,_=[o&&p>1?u-1:u,o&&m>1?d-1:d],v=[o&&p>1?p-1:p,o&&m>1?m-1:m],y=_[0]/v[0],b=_[1]/v[1],x=1/y,S=1/b,C=Math.ceil(x)*2+2,w=Math.ceil(S)*2+2;for(let e=0;e<l;e++){let t=e*s[0];for(let e=0;e<u;e++){let n=t+e*s[1],r=Math.floor(e*x),i=Math.floor(r-C/2);for(let r=0;r<d;r++){let a=n+r*s[2],l=Math.floor(r*S),_=Math.floor(l-w/2);for(let n=0;n<f;n++){let s=0;for(let a=0;a<C;a++){let l=a+i;if(l<0||l>=p)continue;let f=t+l*c[1],h=l*y,v=Math.min(u-1,o?Math.round(h):Math.floor(h));if(e===v)for(let e=0;e<w;e++){let t=e+_;if(t<0||t>=m)continue;let i=f+t*c[2],a=t*b,l=Math.min(d-1,o?Math.round(a):Math.floor(a));r===l&&(s+=g[i+n])}}h[a+n]=s}}}}return n.makeTensorInfo(i.shape,i.dtype,h)}var wM={kernelName:Ln,backendName:`cpu`,kernelFunc:CM};function TM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r;J(i,`reverse`);let o=i.shape.length,s=O(a,i.shape);if(o===0)return TE({inputs:{x:i},backend:n});let c=new Oi(i.shape,i.dtype),l=n.bufferSync(i);for(let e=0;e<c.size;e++){let t=c.indexToLoc(e),n=t.slice();s.forEach(e=>n[e]=i.shape[e]-1-n[e]),c.set(l.get(...n),...t)}return n.makeTensorInfo(c.shape,c.dtype,c.values)}var EM={kernelName:Vn,backendName:`cpu`,kernelFunc:TM},DM={kernelName:Dr,backendName:`cpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,{radians:i,fillValue:a,center:o}=t,s=n,c=re(r.dtype,C(r.shape)),[l,u,d,f]=r.shape,[p,m]=Eh(o,u,d),h=Math.sin(i),g=Math.cos(i),_=s.data.get(r.dataId).values;for(let e=0;e<l;e++){let t=e*d*u*f;for(let e=0;e<u;e++){let n=d*f*e;for(let r=0;r<d;r++){let i=r*f;for(let o=0;o<f;o++){let s=[l,e,r,o],v=s[2],y=s[1],b=(v-p)*g-(y-m)*h,x=(v-p)*h+(y-m)*g;b=Math.round(b+p),x=Math.round(x+m);let S=a;if(typeof a!=`number`&&(S=o===3?255:a[o]),b>=0&&b<d&&x>=0&&x<u){let e=d*f*x,n=b*f;S=_[t+e+n+o]}let C=t+n+i+o;c[C]=S}}}}return{dataId:s.write(c,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}},OM={kernelName:Hn,backendName:`cpu`,kernelFunc:HE(Hn,e=>{let t=Math.floor(e);return e-t<.5?Math.floor(e):e-t>.5?Math.ceil(e):t%2==0?t:t+1})};function kM(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i,updates:a}=t,{shape:o}=r,{sliceRank:s,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=Ff(a,i,o),f=tO(n.bufferSync(i),n.bufferSync(a),o,d,l,c,s,u,0,!0);return n.makeTensorInfo(o,f.dtype,f.values)}var AM={kernelName:Wn,backendName:`cpu`,kernelFunc:kM};function jM(e,t){let n=0,r=e.length,i=0;for(;n<r;)i=Math.floor((n+r)/2),e[i]<t?n=i+1:r=i;return r}function MM(e,t){let n=0,r=e.length,i=0;for(;n<r;)i=Math.floor((n+r)/2),e[i]<=t?n=i+1:r=i;return r}function NM(e,t,n,r,i,a){let o=ie(`int32`,n*i);for(let s=0;s<n;++s){let n=e.slice(s*r,(s+1)*r),c=s*i;for(let e=0;e<i;++e)o[c+e]=a===`left`?jM(n,t[e+c]):MM(n,t[e+c])}return o}function PM(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,l=NM(s,c,i.shape[0],i.shape[1],a.shape[1],o);return n.makeTensorInfo(a.shape,`int32`,l)}var FM={kernelName:Kn,backendName:`cpu`,kernelFunc:PM};function IM(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t;J([r,i,a],`select`);let o=r.shape.length,s=n.data.get(r.dataId).values,c=n.data.get(i.dataId).values,l=n.data.get(a.dataId).values,u=Vi(i.dtype,a.dtype),d=be(C(i.shape),u),f=0,p=o===0||o>1||i.shape.length===1?1:C(i.shape.slice(1));for(let e=0;e<s.length;e++)for(let t=0;t<p;t++)s[e]===1?d[f++]=c[e]:d[f++]=l[e];return n.makeTensorInfo(i.shape,u,d)}var LM={kernelName:qn,backendName:`cpu`,kernelFunc:IM},RM=Mh,zM=Nh,BM={kernelName:Jn,backendName:`cpu`,kernelFunc:HE(Jn,e=>e>=0?zM*e:RM*(Math.exp(e)-1))},VM={kernelName:Zn,backendName:`cpu`,kernelFunc:HE(Zn,e=>e<0?-1:+(e>0))},HM={kernelName:`Sin`,backendName:`cpu`,kernelFunc:HE(`Sin`,e=>Math.sin(e))},UM={kernelName:Xn,backendName:`cpu`,kernelFunc:HE(Xn,e=>Math.sinh(e))},WM=Math.log(1.1920928955078125e-7)+2,GM={kernelName:$n,backendName:`cpu`,kernelFunc:HE($n,e=>{let t=e>-WM,n=e<WM,r=Math.exp(e),i;return i=n?r:t?e:Math.log(1+r),i})};function KM(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r;J([i],`spaceToBatchND`);let s=C(a),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=oM.kernelFunc({inputs:{x:i},backend:n,attrs:{paddings:c,constantValue:0}}),u=Dh(l.shape,a,s,!1),d=Oh(u.length,a.length,!1),f=kh(l.shape,a,s,!1),p=WO({inputs:{x:l},backend:n,attrs:{shape:u}}),m=MD({inputs:{x:p},backend:n,attrs:{perm:d}}),h=WO({inputs:{x:m},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),h}var qM={kernelName:tr,backendName:`cpu`,kernelFunc:KM};function JM(e){let{inputs:t,backend:n}=e,{indices:r,values:i,denseShape:a,defaultValue:o}=t;if(a.shape.length!==1)throw Error(`Dense shape must be a vector, saw:
        ${a.shape}`);if(r.shape.length!==2)throw Error(`Indices must be a matrix, saw:
        ${r.shape}`);if(i.shape.length!==1)throw Error(`Values must be a vector, saw:
        ${i.shape}`);if(o.shape.length!==0)throw Error(`Default value must be a scalar, saw:
        ${o.shape}`);let s=n.data.get(r.dataId).values,c=n.data.get(i.dataId).values,l=n.data.get(a.dataId).values,u=n.data.get(o.dataId).values[0],[d,f,p,m,h]=cO(s,r.shape,r.dtype,c,i.dtype,l,u);return[n.makeTensorInfo(f,r.dtype,d),n.makeTensorInfo([f[0]],i.dtype,p),n.makeTensorInfo([m.length],`bool`,new Uint8Array(m.map(e=>Number(e)))),n.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}var YM={kernelName:ir,backendName:`cpu`,kernelFunc:JM};function XM(e){let{inputs:t,backend:n}=e,{inputIndices:r,inputShape:i,newShape:a}=t;if(r.shape.length!==2)throw Error(`Input indices should be a matrix but received shape
        ${r.shape}`);if(i.shape.length!==1)throw Error(`Input shape should be a vector but received shape
        ${i.shape}`);if(a.shape.length!==1)throw Error(`Target shape should be a vector but received shape ${a.shape}`);let o=Array.from(n.data.get(i.dataId).values),s=n.data.get(r.dataId).values,c=Array.from(n.data.get(a.dataId).values),[l,u,d]=lO(s,r.shape,r.dtype,o,c);return[n.makeTensorInfo(u,r.dtype,l),n.makeTensorInfo([d.length],a.dtype,new Int32Array(d))]}var ZM={kernelName:ar,backendName:`cpu`,kernelFunc:XM};function QM(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
          ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
          ${a.shape}`);if(i.shape[0]!==a.shape[0])throw Error(`segmentIds and indices should have same size.`);let o=n.data.get(r.dataId).values,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,[l,u]=uO(o,r.shape,r.dtype,s,c,!0);return n.makeTensorInfo(u,r.dtype,l)}var $M={kernelName:or,backendName:`cpu`,kernelFunc:QM};function eN(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
         ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
         ${a.shape}`);if(i.shape[0]!==a.shape[0])throw Error(`segmentIds and indices should have same size.`);let o=n.data.get(r.dataId).values,s=n.data.get(i.dataId).values,c=n.data.get(a.dataId).values,[l,u]=uO(o,r.shape,r.dtype,s,c);return n.makeTensorInfo(u,r.dtype,l)}var tN={kernelName:sr,backendName:`cpu`,kernelFunc:eN};function nN(e){let{inputs:t,backend:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:o}=t,{outputShape:s}=r,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=Ff(a,i,s),p=n.bufferSync(i),m;switch(a.dtype){case`bool`:m=tO(p,n.bufferSync(a),s,f,u,l,c,d,!!n.data.get(o.dataId).values[0],!1);break;case`float32`:{let e=n.bufferSync(a),t=n.data.get(o.dataId).values[0];m=tO(p,e,s,f,u,l,c,d,t,!1);break}case`int32`:{let e=n.bufferSync(a),t=n.data.get(o.dataId).values[0];m=tO(p,e,s,f,u,l,c,d,t,!1);break}case`string`:m=tO(p,n.bufferSync(a),s,f,u,l,c,d,di(n.data.get(o.dataId).values[0]),!1);break;default:throw Error(`Unsupported type ${a.dtype}`)}return n.makeTensorInfo(s,m.dtype,m.values)}var rN={kernelName:cr,backendName:`cpu`,kernelFunc:nN};function iN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=r,s=O(o,i.shape)[0],c=ig(i,a,s),l=Array(i.shape.length).fill(0),u=i.shape.slice();return c.map(e=>{let t=[...u];t[s]=e;let r=oO({inputs:{x:i},backend:n,attrs:{begin:l,size:t}});return l[s]+=e,r})}var aN={kernelName:nr,backendName:`cpu`,kernelFunc:iN},oN={kernelName:ur,backendName:`cpu`,kernelFunc:({inputs:e,backend:t})=>{let{x:n}=e,r=t;J(n,`square`);let i=r.data.get(n.dataId).values,a=new Float32Array(i.length);for(let e=0;e<i.length;++e){let t=i[e];a[e]=t*t}return{dataId:r.write(a,n.shape,n.dtype),shape:n.shape,dtype:n.dtype}}},sN={kernelName:Tr,backendName:`cpu`,kernelFunc:HE(Tr,(e,t)=>isNaN(e)?NaN:e>0?1:t.alpha)};function cN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,end:o,strides:s,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:f}=r;J(i,`stridedSlice`);let{finalShapeSparse:p,finalShape:m,isIdentity:h,sliceDim0:g,isSimpleSlice:_,begin:v,end:y,strides:x}=fh(i.shape,a,o,s,c,l,u,d,f),S;if(h)S=WO({inputs:{x:i},backend:n,attrs:{shape:m}});else if(g||_){b(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let e=$m(v,y,x),t=oO({inputs:{x:i},backend:n,attrs:{begin:v,size:e}});S=WO({inputs:{x:t},backend:n,attrs:{shape:m}}),n.disposeIntermediateTensorInfo(t)}else{let e=_O(p,n.bufferSync(i),x,v);S=n.makeTensorInfo(m,e.dtype,e.values)}return S}var lN={kernelName:fr,backendName:`cpu`,kernelFunc:cN};function uN(e){let{inputs:t,backend:n,attrs:r}=e,{separator:i,nGramWidths:a,leftPad:o,rightPad:s,padWidth:c,preserveShortSequences:l}=r,{data:u,dataSplits:d}=t,f=n.data.get(u.dataId).values,p=n.data.get(d.dataId).values,[m,h]=yO(f,p,i,a,o,s,c,l);return[n.makeTensorInfo([m.length],`string`,m),n.makeTensorInfo(d.shape,`int32`,h)]}var dN={kernelName:pr,backendName:`cpu`,kernelFunc:uN};function fN(e){let{inputs:t,backend:n,attrs:r}=e,{skipEmpty:i}=r,{input:a,delimiter:o}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(a.shape.length!==1)throw Error(`Input must be a vector, got shape: ${a.shape}`);if(o.shape.length!==0)throw Error(`Delimiter must be a scalar, got shape: ${o.shape}`);let s=n.data.get(a.dataId).values,c=n.data.get(o.dataId).values[0],[l,u,d]=xO(s,c,i),f=u.length;return[n.makeTensorInfo([f,2],`int32`,l),n.makeTensorInfo([f],`string`,u),n.makeTensorInfo([2],`int32`,new Int32Array(d))]}var pN={kernelName:mr,backendName:`cpu`,kernelFunc:fN};function mN(e){let{inputs:t,backend:n,attrs:r}=e,{numBuckets:i}=r,{input:a}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(i<=0)throw Error(`Number of buckets must be at least 1`);let o=n.data.get(a.dataId).values,s=SO(o,i);return n.makeTensorInfo(a.shape,`int32`,s)}var hN={kernelName:hr,backendName:`cpu`,kernelFunc:mN},gN={kernelName:`Tan`,backendName:`cpu`,kernelFunc:HE(`Tan`,e=>Math.tan(e))},_N={kernelName:gr,backendName:`cpu`,kernelFunc:HE(gr,e=>Math.tanh(e))};function vN(e){let{inputs:t,backend:n}=e,{tensor:r,indices:i,updates:a}=t,{sliceRank:o,numUpdates:s,sliceSize:c,strides:l,outputSize:u}=Ff(a,i,r.shape),d=n.bufferSync(i),f=n.bufferSync(a),p=n.bufferSync(r),m=tO(d,f,r.shape,u,c,s,o,l,p,!1);return n.makeTensorInfo(r.shape,m.dtype,m.values)}var yN={kernelName:Gn,backendName:`cpu`,kernelFunc:vN};function bN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reps:a}=r;J(i,`tile`);let o=EO(n.bufferSync(i),a);return n.makeTensorInfo(o.shape,o.dtype,o.values)}var xN={kernelName:_r,backendName:`cpu`,kernelFunc:bN};function SN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{k:a,sorted:o}=r;J(i,`topk`);let s=n.data.get(i.dataId).values,[c,l]=kO(s,i.shape,i.dtype,a,o);return[n.makeTensorInfo(c.shape,c.dtype,c.values),n.makeTensorInfo(l.shape,l.dtype,l.values)]}var CN={kernelName:vr,backendName:`cpu`,kernelFunc:SN};function wN(e){let{inputs:t,attrs:n,backend:r}=e,{image:i,transforms:a}=t,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=n,[u,d,f,p]=i.shape,[m,h]=l??[d,f],g=[u,m,h,p],_=k(i.shape),v=_[0],y=_[1],b=_[2],x=k(g),S=x[0],w=x[1],T=x[2],E=re(i.dtype,C(g));E.fill(c);let D=r.data.get(i.dataId).values,ee=r.data.get(a.dataId).values;for(let e=0;e<u;++e){let t=a.shape[0]===1?ee:ee.subarray(e*8,e*8+8);for(let n=0;n<m;++n)for(let r=0;r<h;++r)for(let i=0;i<p;++i){let a,l=t[6]*r+t[7]*n+1;if(l===0)continue;let u=(t[0]*r+t[1]*n+t[2])/l,p=(t[3]*r+t[4]*n+t[5])/l,m=EN(u,f,s),h=EN(p,d,s);switch(o){case`nearest`:a=MN(D,d,f,v,y,b,e,h,m,i,c);break;case`bilinear`:a=NN(D,d,f,v,y,b,e,h,m,i,c);break;default:throw Error(`Error in Transform: Expect 'nearest' or 'bilinear', but got ${o}`)}let g=e*S+n*w+r*T+i;E[g]=a}return r.makeTensorInfo(g,i.dtype,E)}return{dataId:r.write(E,g,i.dtype),shape:i.shape,dtype:i.dtype}}var TN={kernelName:yr,backendName:`cpu`,kernelFunc:wN};function EN(e,t,n){switch(n){case`reflect`:return DN(e,t);case`wrap`:return ON(e,t);case`nearest`:return AN(e,t);default:return kN(e,t)}}function DN(e,t){let n=e;if(n<0){if(t<=1)n=0;else{let e=2*t;n<e&&(n=e*Math.trunc(-n/e)+n),n=n<-t?n+e:-n-1}}else if(n>t-1){if(t<=1)n=0;else{let e=2*t;n-=e*Math.trunc(n/e),n>=t&&(n=e-n-1)}}return g(0,n,t-1)}function ON(e,t){let n=e;if(n<0){if(t<=1)n=0;else{let e=t-1;n+=t*(Math.trunc(-n/e)+1)}}else if(n>t-1){if(t<=1)n=0;else{let e=t-1;n-=t*Math.trunc(n/e)}}return g(0,n,t-1)}function kN(e,t){return e}function AN(e,t){return g(0,e,t-1)}function jN(e,t,n,r,i,a,o,s,c,l,u){let d=o*r+s*i+c*a+l;return 0<=s&&s<t&&0<=c&&c<n?e[d]:u}function MN(e,t,n,r,i,a,o,s,c,l,u){return jN(e,t,n,r,i,a,o,Math.round(s),Math.round(c),l,u)}function NN(e,t,n,r,i,a,o,s,c,l,u){let d=Math.floor(s),f=Math.floor(c),p=d+1,m=f+1,h=(m-c)*jN(e,t,n,r,i,a,o,d,f,l,u)+(c-f)*jN(e,t,n,r,i,a,o,d,m,l,u),g=(m-c)*jN(e,t,n,r,i,a,o,p,f,l,u)+(c-f)*jN(e,t,n,r,i,a,o,p,m,l,u);return(p-s)*h+(s-d)*g}function PN(e){let{inputs:t,attrs:n,backend:r}=e,{axis:i}=n,{x:a}=t;J(a,`unique`);let o=r.data.get(a.dataId).values,{outputValues:s,outputShape:c,indices:l}=AO(o,i,a.shape,a.dtype);return[r.makeTensorInfo(c,a.dtype,s),r.makeTensorInfo([l.length],`int32`,l)]}var FN={kernelName:xr,backendName:`cpu`,kernelFunc:PN};function IN(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i.shape.length,s=i.shape[a],c=Array(o-1),l=0;for(let e=0;e<o;e++)e!==a&&(c[l++]=i.shape[e]);let u=Array(o).fill(0),d=i.shape.slice();d[a]=1;let f=Array(s);for(let e=0;e<f.length;e++){u[a]=e;let t=oO({inputs:{x:i},backend:n,attrs:{begin:u,size:d}});f[e]=WO({inputs:{x:t},backend:n,attrs:{shape:c}}),n.disposeIntermediateTensorInfo(t)}return f}var LN={kernelName:Sr,backendName:`cpu`,kernelFunc:IN};function RN(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,segmentIds:a}=t,{numSegments:o}=r;J(i,`unsortedSegmentSum`);let s=i.shape.length,c=a.shape.length,l=[],u=[],d=s-c,f=a;for(let e=0;e<d;++e){let t=FA({inputs:{input:f},backend:n,attrs:{dim:e+1}});f=t,u.push(t)}for(let e=0;e<o;++e){let t=oi(e,`int32`),r=n.makeTensorInfo([],`int32`,t),a=JE({inputs:{a:r,b:f},backend:n}),o=AE({inputs:{x:a},backend:n,attrs:{dtype:`float32`}}),s=wD({inputs:{a:o,b:i},backend:n}),c=SA({inputs:{x:s},backend:n,attrs:{axis:0,keepDims:!1}});l.push(c),u.push(r),u.push(a),u.push(o),u.push(s),u.push(c)}let p=rM({inputs:l,backend:n,attrs:{axis:0}});return u.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}var zN=[YO,bE,XO,ZO,IE,$O,tk,rk,ak,sk,ck,lk,uk,dk,fk,vk,bk,Sk,wk,qO,Ek,Ok,Ak,BE,Mk,jE,GE,Nk,CE,Pk,Rk,Bk,Hk,Wk,Kk,Jk,Xk,Zk,Qk,eA,nA,iA,oA,cA,uA,fA,mA,gA,_A,vA,yA,xA,TA,NO,DA,YE,PA,QE,IA,eD,GA,qA,YA,nD,iD,ZA,$A,tj,rj,cD,uD,EE,aj,Ik,oj,sj,cj,FO,fD,mD,uj,_D,dj,fj,pj,mj,gj,vj,bj,bD,Sj,wj,Ej,Oj,Aj,Mj,Pj,SD,Ij,Lj,Vj,TD,OD,Wj,qj,Xj,AD,Qj,nM,iM,oM,sM,RO,ID,lM,dM,pM,hM,OE,RA,gM,BO,HO,GO,vM,bM,SM,wM,EM,DM,OM,eO,AM,FM,LM,BM,iO,VM,HM,UM,sO,zj,GM,qM,YM,ZM,$M,tN,rN,aN,fO,oN,mO,gO,sN,lN,dN,pN,hN,TO,CA,gN,_N,yN,xN,CN,TN,ND,FN,LN,{kernelName:Cr,backendName:`cpu`,kernelFunc:RN},eM];for(let e of zN)Rr(e);var BN={},VN={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function HN(e,t){BN[e]=t}function UN(e,t){if(!(e in BN)||t!=null){let n=GN(e,t);if(n!==null)BN[e]=n;else return console.log(`Could not get context for WebGL version`,e),null}let n=BN[e];return n==null||n.isContextLost()?(delete BN[e],UN(e)):(n.disable(n.DEPTH_TEST),n.disable(n.STENCIL_TEST),n.disable(n.BLEND),n.disable(n.DITHER),n.disable(n.POLYGON_OFFSET_FILL),n.disable(n.SAMPLE_COVERAGE),n.enable(n.SCISSOR_TEST),n.enable(n.CULL_FACE),n.cullFace(n.BACK),BN[e])}function WN(e){if(!A().getBool(`IS_SAFARI`)&&typeof OffscreenCanvas<`u`&&e===2)return new OffscreenCanvas(300,150);if(typeof document<`u`)return document.createElement(`canvas`);throw Error(`Cannot create a canvas in this context`)}function GN(e,t){if(e!==1&&e!==2)throw Error(`Cannot get WebGL rendering context, WebGL is disabled.`);let n=t??WN(e);return n.addEventListener(`webglcontextlost`,t=>{t.preventDefault(),delete BN[e]},!1),A().getBool(`SOFTWARE_WEBGL_ENABLED`)&&(VN.failIfMajorPerformanceCaveat=!1),e===1?n.getContext(`webgl`,VN)||n.getContext(`experimental-webgl`,VN):n.getContext(`webgl2`,VN)}var KN;(function(e){e[e.DENSE=0]=`DENSE`,e[e.SHARED_BATCH=1]=`SHARED_BATCH`})(KN||={});var qN;(function(e){e[e.RENDER=0]=`RENDER`,e[e.UPLOAD=1]=`UPLOAD`,e[e.PIXELS=2]=`PIXELS`,e[e.DOWNLOAD=3]=`DOWNLOAD`})(qN||={});var JN;(function(e){e[e.UNPACKED_FLOAT16=0]=`UNPACKED_FLOAT16`,e[e.UNPACKED_FLOAT32=1]=`UNPACKED_FLOAT32`,e[e.PACKED_4X1_UNSIGNED_BYTE=2]=`PACKED_4X1_UNSIGNED_BYTE`,e[e.PACKED_2X2_FLOAT32=3]=`PACKED_2X2_FLOAT32`,e[e.PACKED_2X2_FLOAT16=4]=`PACKED_2X2_FLOAT16`})(JN||={});function YN(e,t){return[t,e]}function XN(e,t){return e*t}function ZN(e){let t=C(e);return E(Math.ceil(t/4))}function QN(e,t){return[Math.max(1,Math.ceil(t/2)),Math.max(1,Math.ceil(e/2))]}function $N(e,t){let[n,r]=QN(e,t);return n*r*4}function eP(e,t){let n=e,r,i,a,o,s,c,l,u,d,f;return A().getNumber(`WEBGL_VERSION`)===2?(r=n.R32F,i=n.R16F,a=n.RGBA16F,o=n.RGBA32F,s=n.RED,l=4,u=1,d=n.HALF_FLOAT,f=n.FLOAT,c=n.RGBA8):(r=e.RGBA,i=e.RGBA,a=e.RGBA,o=n.RGBA,s=e.RGBA,l=4,u=4,d=t==null?null:t.HALF_FLOAT_OES,f=e.FLOAT,c=e.RGBA),{internalFormatFloat:r,internalFormatHalfFloat:i,internalFormatPackedHalfFloat:a,internalFormatPackedFloat:o,textureFormatFloat:s,downloadTextureFormat:c,downloadUnpackNumChannels:l,defaultNumChannels:u,textureTypeHalfFloat:d,textureTypeFloat:f}}function Y(e,t){let n=t();return A().getBool(`DEBUG`)&&tP(e),n}function tP(e){let t=e.getError();if(t!==e.NO_ERROR)throw Error(`WebGL Error: `+aP(e,t))}var nP=5.96e-8,rP=65504;function iP(e){return!!(A().getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)||e===0||nP<Math.abs(e)&&Math.abs(e)<rP)}function aP(e,t){switch(t){case e.NO_ERROR:return`NO_ERROR`;case e.INVALID_ENUM:return`INVALID_ENUM`;case e.INVALID_VALUE:return`INVALID_VALUE`;case e.INVALID_OPERATION:return`INVALID_OPERATION`;case e.INVALID_FRAMEBUFFER_OPERATION:return`INVALID_FRAMEBUFFER_OPERATION`;case e.OUT_OF_MEMORY:return`OUT_OF_MEMORY`;case e.CONTEXT_LOST_WEBGL:return`CONTEXT_LOST_WEBGL`;default:return`Unknown error code ${t}`}}function oP(e,t){return OP(e,()=>e.getExtension(t),`Extension "`+t+`" not supported on this browser.`)}function sP(e,t){let n=OP(e,()=>e.createShader(e.VERTEX_SHADER),`Unable to create vertex WebGLShader.`);if(Y(e,()=>e.shaderSource(n,t)),Y(e,()=>e.compileShader(n)),e.getShaderParameter(n,e.COMPILE_STATUS)===!1)throw console.log(e.getShaderInfoLog(n)),Error(`Failed to compile vertex shader.`);return n}function cP(e,t){let n=OP(e,()=>e.createShader(e.FRAGMENT_SHADER),`Unable to create fragment WebGLShader.`);if(Y(e,()=>e.shaderSource(n,t)),Y(e,()=>e.compileShader(n)),A().get(`ENGINE_COMPILE_ONLY`))return n;if(e.getShaderParameter(n,e.COMPILE_STATUS)===!1)throw uP(t,e.getShaderInfoLog(n)),Error(`Failed to compile fragment shader.`);return n}var lP=/ERROR: [0-9]+:([0-9]+):/g;function uP(e,t){let n=lP.exec(t);if(n==null){console.log(`Couldn't parse line number in error: ${t}`),console.log(e);return}let r=+n[1],i=e.split(`
`),a=i.length.toString().length+2,o=i.map((e,t)=>D((t+1).toString(),a)+e),s=0;for(let e=0;e<o.length;e++)s=Math.max(o[e].length,s);let c=o.slice(0,r-1),l=o.slice(r-1,r),u=o.slice(r);console.log(c.join(`
`)),console.log(t.split(`
`)[0]),console.log(`%c ${D(l[0],s)}`,`border:1px solid red; background-color:#e3d2d2; color:#a61717`),console.log(u.join(`
`))}function dP(e){return OP(e,()=>e.createProgram(),`Unable to create WebGLProgram.`)}function fP(e,t){if(Y(e,()=>e.linkProgram(t)),!A().get(`ENGINE_COMPILE_ONLY`)&&e.getProgramParameter(t,e.LINK_STATUS)===!1)throw console.log(e.getProgramInfoLog(t)),Error(`Failed to link vertex and fragment shaders.`)}function pP(e,t){if(Y(e,()=>e.validateProgram(t)),e.getProgramParameter(t,e.VALIDATE_STATUS)===!1)throw console.log(e.getProgramInfoLog(t)),Error(`Shader program validation failed.`)}function mP(e,t){let n=OP(e,()=>e.createBuffer(),`Unable to create WebGLBuffer`);return Y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,n)),Y(e,()=>e.bufferData(e.ARRAY_BUFFER,t,e.STATIC_DRAW)),n}function hP(e,t){let n=OP(e,()=>e.createBuffer(),`Unable to create WebGLBuffer`);return Y(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,n)),Y(e,()=>e.bufferData(e.ELEMENT_ARRAY_BUFFER,t,e.STATIC_DRAW)),n}function gP(e){return OP(e,()=>e.createTexture(),`Unable to create WebGLTexture.`)}function _P(e,t){let n=A().getNumber(`WEBGL_MAX_TEXTURE_SIZE`);if(e<=0||t<=0){let n=`[${e}x${t}]`;throw Error(`Requested texture size `+n+` is invalid.`)}if(e>n||t>n){let r=`[${e}x${t}]`,i=`[${n}x${n}]`;throw Error(`Requested texture size `+r+` greater than WebGL maximum on this browser / GPU `+i+`.`)}}function vP(e){return OP(e,()=>e.createFramebuffer(),`Unable to create WebGLFramebuffer.`)}function yP(e,t,n,r,i,a,o){let s=e.getAttribLocation(t,n);return s!==-1&&(Y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,r)),Y(e,()=>e.vertexAttribPointer(s,i,e.FLOAT,!1,a,o)),Y(e,()=>e.enableVertexAttribArray(s)),!0)}function bP(e,t,n){kP(e,n),Y(e,()=>e.activeTexture(e.TEXTURE0+n)),Y(e,()=>e.bindTexture(e.TEXTURE_2D,t))}function xP(e,t,n){return OP(e,()=>e.getUniformLocation(t,n),`uniform "`+n+`" not present in program.`)}function SP(e,t,n){return e.getUniformLocation(t,n)}function CP(e,t,n,r){Y(e,()=>bP(e,t,r)),Y(e,()=>e.uniform1i(n,r))}function wP(e,t,n){Y(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,n)),Y(e,()=>e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,t,0))}function TP(e,t){Y(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,t)),Y(e,()=>e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,null,0))}function EP(e){let t=e.checkFramebufferStatus(e.FRAMEBUFFER);if(t!==e.FRAMEBUFFER_COMPLETE)throw Error(`Error binding framebuffer: `+DP(e,t))}function DP(e,t){switch(t){case e.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return`FRAMEBUFFER_INCOMPLETE_ATTACHMENT`;case e.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return`FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT`;case e.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return`FRAMEBUFFER_INCOMPLETE_DIMENSIONS`;case e.FRAMEBUFFER_UNSUPPORTED:return`FRAMEBUFFER_UNSUPPORTED`;default:return`unknown error ${t}`}}function OP(e,t,n){let r=Y(e,()=>t());if(r==null)throw Error(n);return r}function kP(e,t){let n=e.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,r=t+e.TEXTURE0;if(r<e.TEXTURE0||r>n){let e=`[gl.TEXTURE0, gl.TEXTURE${n}]`;throw Error(`textureUnit must be in ${e}.`)}}function AP(e,t=2){return C(e.slice(0,e.length-t))}function jP(e){if(e.length===0)throw Error(`Cannot get rows and columns of an empty shape array.`);return[e.length>1?e[e.length-2]:1,e[e.length-1]]}function MP(e){let t=[1,1,1];return e.length===0||e.length===1&&e[0]===1||(t=[AP(e),...jP(e)]),t}function NP(e,t=!1){let n=A().getNumber(`WEBGL_MAX_TEXTURE_SIZE`),r=A().getNumber(`WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE`);r===1/0&&A().getBool(`WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE`)&&(r=n/2),t&&(n*=2,r*=2,e=e.map((t,n)=>n>=e.length-2?_(e[n]):e[n]),e.length===1&&(e=[2,e[0]])),e.length!==2&&(e=ne(e).newShape);let i=C(e),a=null;e.length<=1&&i<=n?a=[1,i]:e.length===2&&e[0]<=n&&e[1]<=n?a=e:e.length===3&&e[0]*e[1]<=n&&e[2]<=n?a=[e[0]*e[1],e[2]]:e.length===3&&e[0]<=n&&e[1]*e[2]<=n?a=[e[0],e[1]*e[2]]:e.length===4&&e[0]*e[1]*e[2]<=n&&e[3]<=n?a=[e[0]*e[1]*e[2],e[3]]:e.length===4&&e[0]<=n&&e[1]*e[2]*e[3]<=n&&(a=[e[0],e[1]*e[2]*e[3]]);let o=a!=null&&Math.max(...a)>r&&Math.min(...a)<=(t?2:1)&&Math.min(...a)>0;if(a==null||o){if(t){let t=AP(e),n=2,r=2;e.length&&([n,r]=jP(e)),i=n/2*t*(r/2),a=E(i).map(e=>e*2)}else a=E(i)}return a}function PP(e){return e%2==0}function FP(e,t){if(e=e.slice(-2),t=t.slice(-2),w(e,t)||!e.length||!t.length||e[0]===0||e[1]===0||t[0]===0||t[1]===0)return!0;if(e.length!==t.length){let n=e[e.length-1],r=t[t.length-1];if(n===r||PP(n)&&PP(r)&&(e[0]===1||t[0]===1))return!0}return e[1]===t[1]&&PP(e[0])&&PP(t[0])}var IP,LP;function RP(e){if(IP==null){let t=UN(e);IP=t.getParameter(t.MAX_TEXTURE_SIZE)}return IP}function zP(e){if(LP==null){let t=UN(e);LP=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,LP)}function BP(e){if(e===0)return 0;let t,n=UN(e);return t=VP(n,`EXT_disjoint_timer_query_webgl2`)&&e===2?2:+!!VP(n,`EXT_disjoint_timer_query`),t}function VP(e,t){return e.getExtension(t)!=null}function HP(e){try{if(UN(e)!=null)return!0}catch(e){return console.log(`Error when getting WebGL context: `,e),!1}return!1}function UP(e){if(e===0)return!1;let t=UN(e);if(e===1){if(!VP(t,`OES_texture_float`))return!1}else if(!VP(t,`EXT_color_buffer_float`))return!1;return GP(t)}function WP(e){if(e===0)return!1;let t=UN(e);if(e===1){if(!VP(t,`OES_texture_float`)||!VP(t,`WEBGL_color_buffer_float`))return!1}else{if(VP(t,`EXT_color_buffer_float`))return GP(t);let e=`EXT_color_buffer_half_float`;return VP(t,e)?KP(t,t.getExtension(e)):!1}return GP(t)}function GP(e){let t=eP(e),n=e.createTexture();e.bindTexture(e.TEXTURE_2D,n),e.texImage2D(e.TEXTURE_2D,0,t.internalFormatFloat,1,1,0,t.textureFormatFloat,t.textureTypeFloat,null);let r=e.createFramebuffer();e.bindFramebuffer(e.FRAMEBUFFER,r),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,n,0);let i=e.checkFramebufferStatus(e.FRAMEBUFFER)===e.FRAMEBUFFER_COMPLETE;return e.bindTexture(e.TEXTURE_2D,null),e.bindFramebuffer(e.FRAMEBUFFER,null),e.deleteTexture(n),e.deleteFramebuffer(r),i}function KP(e,t){let n=eP(e,t),r=e.createTexture();e.bindTexture(e.TEXTURE_2D,r),e.texImage2D(e.TEXTURE_2D,0,n.internalFormatHalfFloat,1,1,0,n.textureFormatFloat,n.textureTypeHalfFloat,null);let i=e.createFramebuffer();e.bindFramebuffer(e.FRAMEBUFFER,i),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,r,0);let a=e.checkFramebufferStatus(e.FRAMEBUFFER)===e.FRAMEBUFFER_COMPLETE;return e.bindTexture(e.TEXTURE_2D,null),e.bindFramebuffer(e.FRAMEBUFFER,null),e.deleteTexture(r),e.deleteFramebuffer(i),a}function qP(e){return e===2&&UN(e).fenceSync!=null}function JP(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{e!=null&&b(e.dtype!==`complex64`,()=>`${t} does not support complex64 tensors in the WebGL backend.`)})}var X=A();X.registerFlag(`HAS_WEBGL`,()=>X.getNumber(`WEBGL_VERSION`)>0),X.registerFlag(`WEBGL_VERSION`,()=>HP(2)?2:+!!HP(1)),X.registerFlag(`WEBGL_CHECK_NUMERICAL_PROBLEMS`,()=>!1),X.registerFlag(`WEBGL_BUFFER_SUPPORTED`,()=>X.get(`WEBGL_VERSION`)===2),X.registerFlag(`WEBGL_CPU_FORWARD`,()=>!0),X.registerFlag(`WEBGL_FORCE_F16_TEXTURES`,()=>!1),X.registerFlag(`WEBGL_PACK`,()=>X.getBool(`HAS_WEBGL`)),X.registerFlag(`WEBGL_PACK_NORMALIZATION`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_CLIP`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_DEPTHWISECONV`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_BINARY_OPERATIONS`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_UNARY_OPERATIONS`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_ARRAY_OPERATIONS`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_IMAGE_OPERATIONS`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_REDUCE`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_LAZILY_UNPACK`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_CONV_IM2COL`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_PACK_CONV2DTRANSPOSE`,()=>X.getBool(`WEBGL_PACK`)),X.registerFlag(`WEBGL_MAX_TEXTURE_SIZE`,()=>RP(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_MAX_TEXTURES_IN_SHADER`,()=>zP(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`,()=>{let e=X.getNumber(`WEBGL_VERSION`);return e===0?0:BP(e)}),X.registerFlag(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`,()=>X.getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)>0&&!ra()),X.registerFlag(`WEBGL_RENDER_FLOAT32_CAPABLE`,()=>UP(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_RENDER_FLOAT32_ENABLED`,()=>!X.getBool(`WEBGL_FORCE_F16_TEXTURES`)&&X.getBool(`WEBGL_RENDER_FLOAT32_CAPABLE`)),X.registerFlag(`WEBGL_DOWNLOAD_FLOAT_ENABLED`,()=>WP(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_FENCE_API_ENABLED`,()=>qP(X.getNumber(`WEBGL_VERSION`))),X.registerFlag(`WEBGL_SIZE_UPLOAD_UNIFORM`,()=>X.getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)?4:0),X.registerFlag(`WEBGL_DELETE_TEXTURE_THRESHOLD`,()=>-1,e=>{if(typeof e!=`number`)throw Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be a number but got ${e}.`);if(e<0&&e!==-1)throw Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${e}.`)}),X.registerFlag(`WEBGL_FLUSH_THRESHOLD`,()=>ra()?1:-1,e=>{if(typeof e!=`number`)throw Error(`WEBGL_FLUSH_THRESHOLD must be a number but got ${e}.`);if(e<0&&e!==-1)throw Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${e}.`)}),X.registerFlag(`CPU_HANDOFF_SIZE_THRESHOLD`,()=>128),X.registerFlag(`WEBGL_USE_SHAPES_UNIFORMS`,()=>!1),X.registerFlag(`TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD`,()=>1e5),X.registerFlag(`TOPK_K_CPU_HANDOFF_THRESHOLD`,()=>128),X.registerFlag(`WEBGL_EXP_CONV`,()=>!1),X.registerFlag(`SOFTWARE_WEBGL_ENABLED`,()=>X.getBool(`IS_TEST`)),X.registerFlag(`WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE`,()=>1/0),X.registerFlag(`WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE`,()=>!1),X.registerFlag(`WEBGL2_ISNAN_CUSTOM`,()=>!1),X.registerFlag(`ENGINE_COMPILE_ONLY`,()=>!1);function YP(){let e,t,n,r,i,a,o,s,c,l;return A().getNumber(`WEBGL_VERSION`)===2?(e=`#version 300 es`,t=`in`,n=`out`,r=`in`,i=`texture`,a=`outputColor`,o=`out vec4 outputColor;`,s=A().getBool(`WEBGL2_ISNAN_CUSTOM`)?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:``,c=``,l=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(e=``,t=`attribute`,n=`varying`,r=`varying`,i=`texture2D`,a=`gl_FragColor`,o=``,s=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,c=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,l=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:e,attribute:t,varyingVs:n,varyingFs:r,texture2D:i,output:a,defineOutput:o,defineSpecialNaN:s,defineSpecialInf:c,defineRound:l}}function XP(e,t,n=`index`){let r=k(t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / ${t}`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * ${t}`:`index -= ${e[i]} * ${t}`};`).join(``)}function ZP(e,t,n=`index`){let r=k(t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / outShapeStrides[${i}]`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * outShapeStrides[${i}]`:`index -= ${e[i]} * outShapeStrides[${i}]`};`).join(``)}function QP(e,t){let n=e.length,r=e.map(e=>`${t}[${e}]`),i=Array(n-1);i[n-2]=r[n-1];for(let e=n-3;e>=0;--e)i[e]=`(${i[e+1]} * ${r[e+1]})`;return i}function $P(e,t,n=`index`){let r=QP(e.map((e,t)=>t),t);return r.map((t,i)=>`${`int ${e[i]} = ${n} / ${r[i]}`}; ${i===r.length-1?`int ${e[i+1]} = ${n} - ${e[i]} * ${r[i]}`:`index -= ${e[i]} * ${r[i]}`};`).join(``)}function eF(e){let t=k(e).map(e=>e.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${t[0]} + coords.y * ${t[1]} + coords.z;
  }
`}function tF(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}var nF=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`,{getBroadcastDims:rF}=xg;function iF(e,t,n){let r=[];if(e.forEach(e=>{let t=C(e.shapeInfo.logicalShape);if(e.shapeInfo.isUniform?r.push(`uniform float ${e.name}${t>1?`[${t}]`:``};`):(r.push(`uniform sampler2D ${e.name};`),r.push(`uniform int offset${e.name};`)),n.enableShapeUniforms){let{uniformShape:t}=KF(n.packedInputs,e.shapeInfo.logicalShape,e.shapeInfo.texShape);switch(t.length){case 1:r.push(`uniform int ${e.name}Shape;`);break;case 2:r.push(`uniform ivec2 ${e.name}Shape;`);break;case 3:r.push(`uniform ivec3 ${e.name}Shape;`);break;case 4:r.push(`uniform ivec4 ${e.name}Shape;`)}r.push(`uniform ivec2 ${e.name}TexShape;`)}}),n.enableShapeUniforms){switch(t.logicalShape.length){case 1:r.push(`uniform int outShape;`);break;case 2:r.push(`uniform ivec2 outShape;`),r.push(`uniform int outShapeStrides;`);break;case 3:r.push(`uniform ivec3 outShape;`),r.push(`uniform ivec2 outShapeStrides;`);break;case 4:r.push(`uniform ivec4 outShape;`),r.push(`uniform ivec3 outShapeStrides;`)}r.push(`uniform ivec2 outTexShape;`)}n.customUniforms&&n.customUniforms.forEach(e=>{r.push(`uniform ${e.type} ${e.name}${e.arrayIndex?`[${e.arrayIndex}]`:``};`)});let i=r.join(`
`),a=e.map(e=>sF(e,t,n.packedInputs,n.enableShapeUniforms)).join(`
`),o=t.texShape,s=YP(),c=uF(s),l,u,d=pF(s);return t.isPacked?(l=cF(t.logicalShape,o,n.enableShapeUniforms),u=fF(s)):(l=lF(t.logicalShape,o,n.enableShapeUniforms),u=dF(s)),n.packedInputs&&(d+=_F),[d,c,u,i,l,a,n.userCode].join(`
`)}function aF(e,t=!1){let n=e.shapeInfo.logicalShape;switch(n.length){case 0:return jF(e,t);case 1:return NF(e,t);case 2:return FF(e,t);case 3:return LF(e,t);case 4:return zF(e,t);case 5:return BF(e);case 6:return VF(e);default:throw Error(`${n.length}-D input sampling is not yet supported`)}}function oF(e,t){switch(e.shapeInfo.logicalShape.length){case 0:return AF(e);case 1:return MF(e,t);case 2:return PF(e,t);case 3:return IF(e,t);default:return RF(e,t)}}function sF(e,t,n=!1,r){let i=``;i+=n?oF(e,r):aF(e,r);let a=e.shapeInfo.logicalShape,o=t.logicalShape;return a.length<=o.length&&(i+=n?UF(e,t):WF(e,t)),i}function cF(e,t,n){switch(e.length){case 0:return vF();case 1:return yF(e,t,n);case 2:return DF(e,t,n);case 3:return xF(e,t,n);default:return CF(e,t,n)}}function lF(e,t,n){switch(e.length){case 0:return vF();case 1:return bF(e,t,n);case 2:return OF(e,t,n);case 3:return SF(e,t,n);case 4:return wF(e,t,n);case 5:return TF(e,t);case 6:return EF(e,t);default:throw Error(`${e.length}-D output sampling is not yet supported`)}}function uF(e){return`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${e.texture2D}(textureSampler, uv).r;
    }
  `}function dF(e){return`
    void setOutput(float val) {
      ${e.output} = vec4(val, 0, 0, 0);
    }
  `}function fF(e){return`
    void setOutput(vec4 val) {
      ${e.output} = val;
    }
  `}function pF(e){return`${e.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${e.varyingFs} vec2 resultUV;
    ${e.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${e.defineSpecialNaN}
    ${e.defineSpecialInf}
    ${e.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${mF}
    ${hF}
    ${gF}
  `}var mF=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,hF=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,gF=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,_F=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function vF(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function yF(e,t,n){let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];return r[0]===1?n?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${r[1]}.0);
      }
    `:r[1]===1?n?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${r[0]}.0);
      }
    `:n?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      return 2 * (resTexRC.x * ${r[1]} + resTexRC.y);
    }
  `}function bF(e,t,n){return t[0]===1?n?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${t[1]}.0);
      }
    `:t[1]===1?n?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${t[0]}.0);
      }
    `:n?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      return resTexRC.x * ${t[1]} + resTexRC.y;
    }
  `}function xF(e,t,n){if(n)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],i=Math.ceil(e[2]/2),a=i*Math.ceil(e[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      int index = resTexRC.x * ${r[1]} + resTexRC.y;

      int b = index / ${a};
      index -= b * ${a};

      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec3(b, r, c);
    }
  `}function SF(e,t,n){if(n)return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${ZP([`r`,`c`,`d`],e)}
    return ivec3(r, c, d);
  }
`;let r=XP([`r`,`c`,`d`],e);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${r}
      return ivec3(r, c, d);
    }
  `}function CF(e,t,n){if(n)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],i=Math.ceil(e[e.length-1]/2),a=i*Math.ceil(e[e.length-2]/2),o=a,s=``,c=`b, r, c`;for(let t=2;t<e.length-1;t++)o*=e[e.length-t-1],s=`
      int b${t} = index / ${o};
      index -= b${t} * ${o};
    `+s,c=`b${t}, `+c;return`
    ivec${e.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));
      int index = resTexRC.x * ${r[1]} + resTexRC.y;

      ${s}

      int b = index / ${a};
      index -= b * ${a};

      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec${e.length}(${c});
    }
  `}function wF(e,t,n){if(n)return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${ZP([`r`,`c`,`d`,`d2`],e)}
      return ivec4(r, c, d, d2);
    }
  `;let r=XP([`r`,`c`,`d`,`d2`],e);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${r}
      return ivec4(r, c, d, d2);
    }
  `}function TF(e,t){let n=XP([`r`,`c`,`d`,`d2`,`d3`],e);return`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${t[0]},
                             ${t[1]}));

      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${n}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `}function EF(e,t){let n=XP([`r`,`c`,`d`,`d2`,`d3`,`d4`],e);return`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${n}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `}function DF(e,t,n){let r=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];if(w(e,t))return n?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${r[0]}, ${r[1]}));
      }
    `;let i=Math.ceil(e[1]/2);return n?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${r[0]}, ${r[1]}));

      int index = resTexRC.x * ${r[1]} + resTexRC.y;
      int r = 2 * (index / ${i});
      int c = imod(index, ${i}) * 2;

      return ivec2(r, c);
    }
  `}function OF(e,t,n){return w(e,t)?n?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${t[0]}, ${t[1]}));
      }
    `:e[1]===1?n?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:e[0]===1?n?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:n?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      int r = index / ${e[1]};
      int c = index - r * ${e[1]};
      return ivec2(r, c);
    }
  `}function kF(e){return`offset${e}`}function AF(e){let t=e.name;return`
    vec4 ${`get`+t.charAt(0).toUpperCase()+t.slice(1)}() {
      return ${YP().texture2D}(${t}, halfCR);
    }
  `}function jF(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1);if(e.shapeInfo.isUniform)return`float ${r}() {return ${n};}`;let[i,a]=e.shapeInfo.texShape;if(i===1&&a===1)return`
      float ${r}() {
        return sampleTexture(${n}, halfCR);
      }
    `;let o=kF(n);if(t)return`
    float ${r}() {
      vec2 uv = uvFromFlat(${n}TexShape[0], ${n}TexShape[1], ${o});
      return sampleTexture(${n}, uv);
    }
  `;let[s,c]=e.shapeInfo.texShape;return`
    float ${r}() {
      vec2 uv = uvFromFlat(${s}, ${c}, ${o});
      return sampleTexture(${n}, uv);
    }
  `}function MF(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=e.shapeInfo.texShape,a=YP();if(t)return`
    vec4 ${r}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${n}TexShape[0]) / 2.0), ceil(float(${n}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${a.texture2D}(${n}, uv);
    }
  `;let o=[Math.ceil(i[0]/2),Math.ceil(i[1]/2)];return`
    vec4 ${r}(int index) {
      vec2 uv = packedUVfrom1D(
        ${o[0]}, ${o[1]}, index);
      return ${a.texture2D}(${n}, uv);
    }
  `}function NF(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1);if(e.shapeInfo.isUniform)return`
      float ${r}(int index) {
        ${HF(e)}
      }
    `;let i=e.shapeInfo.texShape,a=i[0],o=i[1];if(o===1&&a===1)return`
      float ${r}(int index) {
        return sampleTexture(${n}, halfCR);
      }
    `;let s=kF(n);return o===1?t?`
      float ${r}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${s}) + 0.5) / float(${n}TexShape[0]));
        return sampleTexture(${n}, uv);
      }
    `:`
      float ${r}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${s}) + 0.5) / ${a}.0);
        return sampleTexture(${n}, uv);
      }
    `:a===1?t?`
      float ${r}(int index) {
        vec2 uv = vec2((float(index + ${s}) + 0.5) / float(${n}TexShape[1]), 0.5);
        return sampleTexture(${n}, uv);
      }
    `:`
      float ${r}(int index) {
        vec2 uv = vec2((float(index + ${s}) + 0.5) / ${o}.0, 0.5);
        return sampleTexture(${n}, uv);
      }
    `:t?`
    float ${r}(int index) {
      vec2 uv = uvFromFlat(${n}TexShape[0], ${n}TexShape[1], index + ${s});
      return sampleTexture(${n}, uv);
    }
  `:`
    float ${r}(int index) {
      vec2 uv = uvFromFlat(${a}, ${o}, index + ${s});
      return sampleTexture(${n}, uv);
    }
  `}function PF(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape,o=a[0],s=a[1],c=YP();if(a!=null&&w(n,a))return t?`
      vec4 ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);

        return ${c.texture2D}(${r}, uv);
      }
    `:`
      vec4 ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}.0, ${o}.0);

        return ${c.texture2D}(${r}, uv);
      }
    `;if(t)return`
    vec4 ${i}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${r}TexShape[0]) / 2.0), ceil(float(${r}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${r}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${c.texture2D}(${r}, uv);
    }
  `;let l=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];return`
    vec4 ${i}(int row, int col) {
      vec2 uv = packedUVfrom2D(${Math.ceil(n[1]/2)}, ${l[0]}, ${l[1]}, row, col);
      return ${c.texture2D}(${r}, uv);
    }
  `}function FF(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape;if(a!=null&&w(n,a)){if(t)return`
      float ${i}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `;let e=a[0];return`
    float ${i}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${a[1]}.0, ${e}.0);
      return sampleTexture(${r}, uv);
    }
  `}let{newShape:o,keptDims:s}=ne(n),c=o;if(c.length<n.length)return`
      ${aF(qF(e,c),t)}
      float ${i}(int row, int col) {
        return ${i}(${JF([`row`,`col`],s)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${n[1]}, 1)));
        ${HF(e)}
      }
    `;let l=a[0],u=a[1],d=kF(r);return u===1?t?`
      float ${i}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${r}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${r}TexShape[0]));
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${n[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${l}.0);
      return sampleTexture(${r}, uv);
    }
  `:l===1?t?`
      float ${i}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${r}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${r}TexShape[1]), 0.5);
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${n[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${u}.0, 0.5);
      return sampleTexture(${r}, uv);
    }
  `:t?`
      float ${i}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${r}Shape[1] + col + ${d};
        vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index);
        return sampleTexture(${r}, uv);
      }
    `:`
  float ${i}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${n[1]} + col + ${d};
    vec2 uv = uvFromFlat(${l}, ${u}, index);
    return sampleTexture(${r}, uv);
  }
`}function IF(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape,o=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];if(n[0]===1)return`
        ${oF(qF(e,n.slice(1)),t)}
        vec4 ${i}(int b, int row, int col) {
          return ${i}(${JF([`b`,`row`,`col`],[1,2])});
        }
      `;let s=YP();if(t)return`
    vec4 ${i}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${r}TexShape[0]) / 2.0), ceil(float(${r}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${r}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${r}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${s.texture2D}(${r}, uv);
    }
  `;let c=o[0],l=o[1],u=Math.ceil(n[2]/2);return`
    vec4 ${i}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${c}, ${l}, ${u*Math.ceil(n[1]/2)}, ${u}, b, row, col);
      return ${s.texture2D}(${r}, uv);
    }
  `}function LF(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=n[1]*n[2],o=n[2],{newShape:s,keptDims:c}=ne(n),l=s;if(l.length<n.length)return`
        ${aF(qF(e,l),t)}
        float ${i}(int row, int col, int depth) {
          return ${i}(${JF([`row`,`col`,`depth`],c)});
        }
      `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${a}, ${o}, 1)));
        ${HF(e)}
      }
    `;let u=e.shapeInfo.texShape,d=u[0],f=u[1],p=e.shapeInfo.flatOffset;if(f===a&&p==null)return t?`
      float ${i}(int row, int col, int depth) {
        int stride1 = ${r}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
        float ${i}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${o}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${f}.0, ${d}.0);
          return sampleTexture(${r}, uv);
        }
      `;if(f===o&&p==null)return t?`
      float ${i}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${r}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
    float ${i}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${n[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${f}.0, ${d}.0);
      return sampleTexture(${r}, uv);
    }
  `;let m=kF(r);return t?`
    float ${i}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${r}Shape[1] * ${r}Shape[2];
      int stride1 = ${r}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${m};
      vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index);
      return sampleTexture(${r}, uv);
    }
    `:`
      float ${i}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${a} + col * ${o} + depth + ${m};
        vec2 uv = uvFromFlat(${d}, ${f}, index);
        return sampleTexture(${r}, uv);
      }
  `}function RF(e,t){let n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=YP();if(t)return`
    vec4 ${r}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${n}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${n}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${n}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${n}TexShape[0]) / 2.0), ceil(float(${n}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${i.texture2D}(${n}, uv);
    }
  `;let a=e.shapeInfo.logicalShape,o=a.length,s=e.shapeInfo.texShape,c=[Math.ceil(s[0]/2),Math.ceil(s[1]/2)],l=c[0],u=c[1],d=Math.ceil(a[o-1]/2),f=d*Math.ceil(a[o-2]/2),p=`int b, int row, int col`,m=`b * ${f} + (row / 2) * ${d} + (col / 2)`;for(let e=2;e<o-1;e++)p=`int b${e}, `+p,f*=a[o-e-1],m=`b${e} * ${f} + `+m;return`
    vec4 ${r}(${p}) {
      int index = ${m};
      int texR = index / ${u};
      int texC = index - texR * ${u};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${l});
      return ${i.texture2D}(${n}, uv);
    }
  `}function zF(e,t){let n=e.shapeInfo.logicalShape,r=e.name,i=`get`+r.charAt(0).toUpperCase()+r.slice(1),a=n[3],o=n[2]*a,s=n[1]*o,{newShape:c,keptDims:l}=ne(n);if(c.length<n.length)return`
      ${aF(qF(e,c),t)}
      float ${i}(int row, int col, int depth, int depth2) {
        return ${i}(${JF([`row`,`col`,`depth`,`depth2`],l)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${i}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${s}, ${o}, ${a}, 1)));
        ${HF(e)}
      }
    `;let u=e.shapeInfo.flatOffset,d=e.shapeInfo.texShape,f=d[0],p=d[1],m=`int stride2 = ${r}Shape[3];`,h=`int stride1 = ${r}Shape[2] * stride2;`,g=`int stride0 = ${r}Shape[1] * stride1;`;if(p===s&&u==null)return t?`
      float ${i}(int row, int col, int depth, int depth2) {
        ${m}
        ${h}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${o}, ${a}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${f}.0);
        return sampleTexture(${r}, uv);
      }
    `;if(p===a&&u==null)return t?`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${r}Shape[1] * ${r}Shape[2], ${r}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${r}TexShape[1], ${r}TexShape[0]);
        return sampleTexture(${r}, uv);
      }
    `:`
      float ${i}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${n[1]*n[2]}, ${n[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${f}.0);
        return sampleTexture(${r}, uv);
      }
    `;let _=kF(r);return t?`
    float ${i}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${m}
      ${h}
      ${g}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index + ${_});
      return sampleTexture(${r}, uv);
    }
  `:`
    float ${i}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${s} + col * ${o} +
          depth * ${a} + depth2;
      vec2 uv = uvFromFlat(${f}, ${p}, index + ${_});
      return sampleTexture(${r}, uv);
    }
  `}function BF(e){let t=e.shapeInfo.logicalShape,n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),i=t[4],a=t[3]*i,o=t[2]*a,s=t[1]*o,{newShape:c,keptDims:l}=ne(t);if(c.length<t.length)return`
      ${aF(qF(e,c))}
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        return ${r}(${JF([`row`,`col`,`depth`,`depth2`,`depth3`],l)});
      }
    `;if(e.shapeInfo.isUniform)return`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${s}, ${o}, ${a}, ${i})) +
          depth3;
        ${HF(e)}
      }
    `;let u=e.shapeInfo.flatOffset,d=e.shapeInfo.texShape,f=d[0],p=d[1];return p===s&&u==null?`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${o}, ${a}, ${i}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${f}.0);
        return sampleTexture(${n}, uv);
      }
    `:p===i&&u==null?`
      float ${r}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]},
               ${t[2]*t[3]}, ${t[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${f}.0);
        return sampleTexture(${n}, uv);
      }
    `:`
    float ${r}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${s} + col * ${o} + depth * ${a} +
          depth2 * ${i} + depth3 + ${kF(n)};
      vec2 uv = uvFromFlat(${f}, ${p}, index);
      return sampleTexture(${n}, uv);
    }
  `}function VF(e){let t=e.shapeInfo.logicalShape,n=e.name,r=`get`+n.charAt(0).toUpperCase()+n.slice(1),{newShape:i,keptDims:a}=ne(t);if(i.length<t.length)return`
      ${aF(qF(e,i))}
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${r}(${JF([`row`,`col`,`depth`,`depth2`,`depth3`,`depth4`],a)});
      }
    `;let o=t[5],s=t[4]*o,c=t[3]*s,l=t[2]*c,u=t[1]*l;if(e.shapeInfo.isUniform)return`
      float ${r}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${u}, ${l}, ${c}, ${s})) +
          dot(
            vec2(depth3, depth4),
            vec2(${o}, 1)));
        ${HF(e)}
      }
    `;let d=e.shapeInfo.flatOffset,f=e.shapeInfo.texShape,p=f[0],m=f[1];return m===u&&d==null?`
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${l}, ${c}, ${s}, ${o})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${m}.0, ${p}.0);
        return sampleTexture(${n}, uv);
      }
    `:m===o&&d==null?`
      float ${r}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]*t[4]},
               ${t[2]*t[3]*t[4]},
               ${t[3]*t[4]},
               ${t[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${m}.0, ${p}.0);
        return sampleTexture(${n}, uv);
      }
    `:`
    float ${r}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${u} + col * ${l} + depth * ${c} +
          depth2 * ${s} + depth3 * ${o} + depth4 + ${kF(n)};
      vec2 uv = uvFromFlat(${p}, ${m}, index);
      return sampleTexture(${n}, uv);
    }
  `}function HF(e){let t=e.name,n=C(e.shapeInfo.logicalShape);return n<2?`return ${t};`:`
    for (int i = 0; i < ${n}; i++) {
      if (i == index) {
        return ${t}[i];
      }
    }
  `}function UF(e,t){let n=e.name,r=n.charAt(0).toUpperCase()+n.slice(1),i=`get`+r+`AtOutCoords`,a=e.shapeInfo.logicalShape.length,o=t.logicalShape.length,s=rF(e.shapeInfo.logicalShape,t.logicalShape),c=GF(o),l=o-a,u,d=[`x`,`y`,`z`,`w`,`u`,`v`];u=a===0?``:o<2&&s.length>=1?`coords = 0;`:s.map(e=>`coords.${d[e+l]} = 0;`).join(`
`);let f=``;f=o<2&&a>0?`coords`:e.shapeInfo.logicalShape.map((e,t)=>`coords.${d[t+l]}`).join(`, `);let p=`return outputValue;`,m=C(e.shapeInfo.logicalShape)===1,h=C(t.logicalShape)===1;if(a===1&&!m&&!h)p=`
      return vec4(outputValue.xy, outputValue.xy);
    `;else if(m&&!h)p=o===1?`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:`
        return vec4(outputValue.x);
      `;else if(s.length){let e=a-2,t=a-1;s.indexOf(e)>-1&&s.indexOf(t)>-1?p=`return vec4(outputValue.x);`:s.indexOf(e)>-1?p=`return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);`:s.indexOf(t)>-1&&(p=`return vec4(outputValue.xx, outputValue.zz);`)}return`
    vec4 ${i}() {
      ${c} coords = getOutputCoords();
      ${u}
      vec4 outputValue = get${r}(${f});
      ${p}
    }
  `}function WF(e,t){let n=e.name,r=n.charAt(0).toUpperCase()+n.slice(1),i=`get`+r+`AtOutCoords`,a=t.texShape,o=e.shapeInfo.texShape,s=e.shapeInfo.logicalShape.length,c=t.logicalShape.length;if(!e.shapeInfo.isUniform&&s===c&&e.shapeInfo.flatOffset==null&&w(o,a))return`
      float ${i}() {
        return sampleTexture(${n}, resultUV);
      }
    `;let l=GF(c),u=rF(e.shapeInfo.logicalShape,t.logicalShape),d=c-s,f,p=[`x`,`y`,`z`,`w`,`u`,`v`];f=s===0?``:c<2&&u.length>=1?`coords = 0;`:u.map(e=>`coords.${p[e+d]} = 0;`).join(`
`);let m=``;return m=c<2&&s>0?`coords`:e.shapeInfo.logicalShape.map((e,t)=>`coords.${p[t+d]}`).join(`, `),`
    float ${i}() {
      ${l} coords = getOutputCoords();
      ${f}
      return get${r}(${m});
    }
  `}function GF(e){if(e<=1)return`int`;if(e===2)return`ivec2`;if(e===3)return`ivec3`;if(e===4)return`ivec4`;if(e===5)return`ivec5`;if(e===6)return`ivec6`;throw Error(`GPU for rank ${e} is not yet supported`)}function KF(e,t,n){let{newShape:r,keptDims:i}=ne(t),a=t.length,o=e&&a===3&&t[0]===1,s=o?t.slice(1):r,c=!e&&a>1&&!w(t,n)&&r.length<a||o;return{useSqueezeShape:c,uniformShape:c?s:t,keptDims:i}}function qF(e,t){let n=JSON.parse(JSON.stringify(e));return n.shapeInfo.logicalShape=t,n}function JF(e,t){return t.map(t=>e[t]).join(`, `)}function YF(e,t,n,r){let i=n.map((e,n)=>{let r={logicalShape:e.shape,texShape:e.isUniform?null:e.texData.texShape,isUniform:e.isUniform,isPacked:!e.isUniform&&e.texData.isPacked,flatOffset:null};return e.texData!=null&&e.texData.slice!=null&&e.texData.slice.flatOffset>0&&(r.flatOffset=e.texData.slice.flatOffset),{name:t.variableNames[n],shapeInfo:r}}),a=i.map(e=>e.shapeInfo),o={logicalShape:r.shape,texShape:r.texData.texShape,isUniform:!1,isPacked:r.texData.isPacked,flatOffset:null},s=iF(i,o,t),c=cP(e.gl,s),l=e.createProgram(c);return A().get(`ENGINE_COMPILE_ONLY`)?{program:t,fragmentShader:c,source:s,webGLProgram:l,inShapeInfos:a,outShapeInfo:o,variablesLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:(e.buildVao(l),Object.assign({program:t,fragmentShader:c,source:s,webGLProgram:l,inShapeInfos:a,outShapeInfo:o},XF(e,t,l)))}function XF(e,t,n){let r=[],i=[],a,o,s,c=null,l=null;l=e.getUniformLocation(n,`NAN`,!1),A().getNumber(`WEBGL_VERSION`)===1&&(c=e.getUniformLocation(n,`INFINITY`,!1));for(let i of t.variableNames){let a={name:i,uniform:e.getUniformLocation(n,i,!1),offset:e.getUniformLocation(n,`offset${i}`,!1)};t.enableShapeUniforms&&(a.shape=e.getUniformLocation(n,`${i}Shape`,!1),a.texShape=e.getUniformLocation(n,`${i}TexShape`,!1)),r.push(a)}if(t.enableShapeUniforms&&(a=e.getUniformLocation(n,`outShape`,!1),s=e.getUniformLocation(n,`outShapeStrides`,!1),o=e.getUniformLocation(n,`outTexShape`,!1)),t.customUniforms)for(let r of t.customUniforms)i.push(e.getUniformLocation(n,r.name,!1));return{variablesLocations:r,customUniformLocations:i,infLoc:c,nanLoc:l,outShapeLocation:a,outShapeStridesLocation:s,outTexShapeLocation:o}}function ZF(e,t){if(e.length!==t.length)throw Error(`Binary was compiled with ${e.length} inputs, but was executed with ${t.length} inputs`);e.forEach((e,n)=>{let r=e.logicalShape,i=t[n],a=i.shape;if(!w(r,a))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${r} and ${a} must match`);if(e.isUniform&&i.isUniform)return;let o=e.texShape,s=i.isUniform?null:i.texData.texShape;if(!w(o,s))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${o} and ${s} must match`)})}function QF(e,t,n,r,i){t.program.enableShapeUniforms||(ZF(t.inShapeInfos,n),ZF([t.outShapeInfo],[r]));let a=r.texData.texture,o=r.texData.texShape;r.texData.isPacked?e.setOutputPackedMatrixTexture(a.texture,o[0],o[1]):e.setOutputMatrixTexture(a.texture,o[0],o[1]),e.setProgram(t.webGLProgram),e.bindVertexArray(t.webGLProgram.vao),A().getNumber(`WEBGL_VERSION`)===1&&t.infLoc!==null&&e.gl.uniform1f(t.infLoc,1/0),t.nanLoc!==null&&e.gl.uniform1f(t.nanLoc,NaN);for(let r=0;r<n.length;++r){let i=n[r],{uniform:a,offset:o,shape:s,texShape:c}=t.variablesLocations[r];if(s){let{uniformShape:n}=KF(t.program.packedInputs,i.shape,i.texData.texShape);switch(n.length){case 1:e.gl.uniform1iv(s,new Int32Array(n));break;case 2:e.gl.uniform2iv(s,new Int32Array(n));break;case 3:e.gl.uniform3iv(s,new Int32Array(n));break;case 4:e.gl.uniform4iv(s,new Int32Array(n))}}if(c&&e.gl.uniform2i(c,i.texData.texShape[0],i.texData.texShape[1]),a!=null){if(i.isUniform){if(C(i.shape)<2)e.gl.uniform1f(a,i.uniformValues[0]);else{let t=i.uniformValues;t instanceof Float32Array||(t=new Float32Array(t)),e.gl.uniform1fv(a,t)}continue}i.texData.slice!=null&&o!=null&&e.gl.uniform1i(o,i.texData.slice.flatOffset),e.setInputMatrixTexture(i.texData.texture.texture,a,r)}}let s=t.outShapeLocation;if(s)switch(r.shape.length){case 1:e.gl.uniform1iv(s,new Int32Array(r.shape));break;case 2:e.gl.uniform2iv(s,new Int32Array(r.shape));break;case 3:e.gl.uniform3iv(s,new Int32Array(r.shape));break;case 4:e.gl.uniform4iv(s,new Int32Array(r.shape))}if(t.outShapeStridesLocation){let n=k(r.shape);switch(r.shape.length){case 2:e.gl.uniform1iv(t.outShapeStridesLocation,new Int32Array(n));break;case 3:e.gl.uniform2iv(t.outShapeStridesLocation,new Int32Array(n));break;case 4:e.gl.uniform3iv(t.outShapeStridesLocation,new Int32Array(n))}}if(t.outTexShapeLocation&&e.gl.uniform2i(t.outTexShapeLocation,r.texData.texShape[0],r.texData.texShape[1]),t.program.customUniforms&&i)for(let n=0;n<t.program.customUniforms.length;++n){let r=t.program.customUniforms[n],a=t.customUniformLocations[n],o=i[n];if(r.type===`float`)e.gl.uniform1fv(a,o);else if(r.type===`vec2`)e.gl.uniform2fv(a,o);else if(r.type===`vec3`)e.gl.uniform3fv(a,o);else if(r.type===`vec4`)e.gl.uniform4fv(a,o);else if(r.type===`int`)e.gl.uniform1iv(a,o);else if(r.type===`ivec2`)e.gl.uniform2iv(a,o);else if(r.type===`ivec3`)e.gl.uniform3iv(a,o);else if(r.type===`ivec4`)e.gl.uniform4iv(a,o);else throw Error(`uniform type ${r.type} is not supported yet.`)}e.executeProgram()}function $F(e,t,n){let r=``;t.concat(n).forEach(t=>{let i=t.texData!=null&&t.texData.slice!=null&&t.texData.slice.flatOffset>0;if(e.enableShapeUniforms&&!t.isUniform){let a=t.texData.texShape,{useSqueezeShape:o,uniformShape:s,keptDims:c}=KF(e.packedInputs,t.shape,a),l=``,u=``,d=``;if(s.length===1&&e.packedInputs){let e=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];l=`${e[0]>1}_${e[1]>1}`}else if(s.length===2&&!e.packedInputs)u=`${s[0]>1}_${s[1]>1}`;else if(s.length>2&&!e.packedInputs){let e=k(s);d=`${e[0]===a[1]}_${e[e.length-1]===a[1]}`}let f=t.shape.length,p=s.length===2&&w(t.shape,a),m=C(t.shape)===1,h=Lc(t.shape,n.shape),g=!e.packedInputs&&f===n.shape.length&&w(a,n.texData.texShape),_=e.packedInputs||s.length>2?``:`${a[0]>1}_${a[1]>1}`;r+=`${f}_${g}_${o?c:``}_${s.length}_${m}_${h}_${p}_${l}_${u}_${d}_${_}_${i}`}else{let e=t.isUniform?`uniform`:t.texData.texShape;r+=`${t.shape}_${e}_${i}`}});let i=e.userCode,a=e.constructor.name;return a+=`_`+r+`_`+i+`${A().getNumber(`WEBGL_VERSION`)}`,a}function eI(e){return A().getBool(`WEBGL_USE_SHAPES_UNIFORMS`)&&e<=4}var tI=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=KN.DENSE,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let t=YP();this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?ZP([`r`,`c`,`d`],e):XP([`r`,`c`,`d`],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${t.output} = result;
      }
    `}},nI=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=KN.DENSE,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let t=YP();this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?ZP([`r`,`c`,`d`],e):XP([`r`,`c`,`d`],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${t.output} = result;
      }
    `}},rI=class{constructor(e){this.variableNames=[`A`],this.outTexUsage=qN.DOWNLOAD;let t=YP();this.outputShape=e,this.userCode=`
      ${nF}

      void main() {
        float x = getAAtOutCoords();
        ${t.output} = encode_float(x);
      }
    `}},iI=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=qN.DOWNLOAD;let t=YP();this.outputShape=e,this.userCode=`
      ${nF}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${t.output} = encode_float(x);
      }
    `}},aI={R:0,G:1,B:2,A:3},oI=class{constructor(e,t=!1,n=`RGBA`){this.variableNames=[`A`],this.customUniforms=[{name:`texShape`,type:`ivec2`}];let r=YP();this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length);let i=`result`;t&&(i=`floor(result * 255. + 0.5)`);let a=``;for(let e=0;e<n.length;e++){let t=n[e];a+=`
          if(offset == ${e}) {
            result = values[${aI[t]}];
          }`}this.userCode=`
      ${this.enableShapeUniforms?tF():eF(e)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${n.length});

        flatIndex = idiv(flatIndex, ${n.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${r.texture2D}(A, uv);
          ${a}
        }
        ${r.output} = vec4(${i}, 0., 0., 0.);
      }
    `}},sI=class{constructor(e,t=!1){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:`texShape`,type:`ivec2`}];let n=YP();this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length);let r=``,i=`result`;t&&(i=`floor(result * 255. + 0.5)`);for(let t=0;t<=1;t++)for(let i=0;i<=1;i++){let a=t*2+i;r+=`
          localCoords = coords;
          if(localCoords[2] + ${i} < ${this.enableShapeUniforms?`outShape[2]`:`${e[2]}`}) {
          localCoords[2] += ${i};
          if (localCoords[1] + ${t} < ${this.enableShapeUniforms?`outShape[1]`:`${e[1]}`}) {
            localCoords[1] += ${t};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${n.texture2D}(A, uv);

            if (offset == 0) {
              result[${a}] = values[0];
            } else if (offset == 1) {
              result[${a}] = values[1];
            } else if (offset == 2) {
              result[${a}] = values[2];
            } else {
              result[${a}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?tF():eF(e)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${r}

          ${n.output} = ${i};
        }
    `}};function cI(e){let t=YP();return sP(e,`${t.version}
    precision highp float;
    ${t.attribute} vec3 clipSpacePos;
    ${t.attribute} vec2 uv;
    ${t.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`)}function lI(e){return mP(e,new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]))}function uI(e){return hP(e,new Uint16Array([0,1,2,2,1,3]))}function dI(e,t,n,r,i,a){_P(t,n);let o=gP(e),s=e.TEXTURE_2D;return Y(e,()=>e.bindTexture(s,o)),Y(e,()=>e.texParameteri(s,e.TEXTURE_WRAP_S,e.CLAMP_TO_EDGE)),Y(e,()=>e.texParameteri(s,e.TEXTURE_WRAP_T,e.CLAMP_TO_EDGE)),Y(e,()=>e.texParameteri(s,e.TEXTURE_MIN_FILTER,e.NEAREST)),Y(e,()=>e.texParameteri(s,e.TEXTURE_MAG_FILTER,e.NEAREST)),A().getNumber(`WEBGL_VERSION`)===1?Y(e,()=>e.texImage2D(s,0,r,t,n,0,i,a,null)):Y(e,()=>e.texStorage2D(s,1,r,t,n)),Y(e,()=>e.bindTexture(e.TEXTURE_2D,null)),{texture:o,texShape:[n,t]}}function fI(e){return e.internalFormatFloat}function pI(e,t,n,r){let[i,a]=YN(t,n);return dI(e,i,a,fI(r),r.textureFormatFloat,e.FLOAT)}function mI(e){return e.internalFormatHalfFloat}function hI(e,t,n,r){let[i,a]=YN(t,n);return dI(e,i,a,mI(r),r.textureFormatFloat,r.textureTypeHalfFloat)}function gI(e){return e.downloadTextureFormat}function _I(e,t,n,r){let[i,a]=YN(t,n);return dI(e,i,a,gI(r),e.RGBA,e.UNSIGNED_BYTE)}function vI(e){return e.internalFormatPackedFloat}function yI(e,t,n,r){let[i,a]=QN(t,n);return dI(e,i,a,vI(r),e.RGBA,e.FLOAT)}function bI(e){return e.internalFormatPackedHalfFloat}function xI(e,t,n,r){let[i,a]=QN(t,n);return dI(e,i,a,bI(r),e.RGBA,r.textureTypeHalfFloat)}function SI(e,t,n){return Y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,n)),yP(e,t,`clipSpacePos`,n,3,20,0)&&yP(e,t,`uv`,n,2,20,12)}function CI(e,t,n,r,i,a){Y(e,()=>e.bindTexture(e.TEXTURE_2D,t));let o,s,c;i instanceof Uint8Array?(o=new Uint8Array(n*r*4),s=e.UNSIGNED_BYTE,c=e.RGBA):(o=new Float32Array(n*r*4),s=e.FLOAT,c=a.internalFormatPackedFloat),o.set(i),A().getNumber(`WEBGL_VERSION`)===2?Y(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,n,r,e.RGBA,s,o)):Y(e,()=>e.texImage2D(e.TEXTURE_2D,0,c,n,r,0,e.RGBA,s,o)),Y(e,()=>e.bindTexture(e.TEXTURE_2D,null))}function wI(e,t,n){Y(e,()=>e.bindTexture(e.TEXTURE_2D,t)),n.data instanceof Uint8Array?A().getNumber(`WEBGL_VERSION`)===2?Y(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,n.width,n.height,e.RGBA,e.UNSIGNED_BYTE,n.data)):Y(e,()=>e.texImage2D(e.TEXTURE_2D,0,e.RGBA,n.width,n.height,0,e.RGBA,e.UNSIGNED_BYTE,n.data)):A().getNumber(`WEBGL_VERSION`)===2?Y(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,e.RGBA,e.UNSIGNED_BYTE,n)):Y(e,()=>e.texImage2D(e.TEXTURE_2D,0,e.RGBA,e.RGBA,e.UNSIGNED_BYTE,n)),Y(e,()=>e.bindTexture(e.TEXTURE_2D,null))}function TI(e,t,n,r){let i=e.createBuffer();Y(e,()=>e.bindBuffer(e.PIXEL_PACK_BUFFER,i));let a=16*t*n;return Y(e,()=>e.bufferData(e.PIXEL_PACK_BUFFER,a,e.STREAM_READ)),Y(e,()=>e.readPixels(0,0,n,t,e.RGBA,e.FLOAT,0)),Y(e,()=>e.bindBuffer(e.PIXEL_PACK_BUFFER,null)),i}function EI(e,t,n){let r=e,i=new Float32Array(n);return r.bindBuffer(r.PIXEL_PACK_BUFFER,t),r.getBufferSubData(r.PIXEL_PACK_BUFFER,0,i),r.bindBuffer(r.PIXEL_PACK_BUFFER,null),i}function DI(e,t,n,r){let[i,a]=YN(t,n),o=new Uint8Array(XN(t*n,4));return Y(e,()=>e.readPixels(0,0,i,a,r.downloadTextureFormat,e.UNSIGNED_BYTE,o)),new Float32Array(o.buffer)}function OI(e,t,n,r,i,a,o,s){let c=e,l=new Float32Array($N(a,o));return c.bindBuffer(c.PIXEL_PACK_BUFFER,t),c.getBufferSubData(c.PIXEL_PACK_BUFFER,0,l),c.bindBuffer(c.PIXEL_PACK_BUFFER,null),l}function kI(e,t,n){let r=new Float32Array(t*n*4);return Y(e,()=>e.readPixels(0,0,n,t,e.RGBA,e.FLOAT,r)),r}var AI=class{constructor(e){this.outputTexture=null,this.program=null,this.disposed=!1,this.itemsToPoll=[];let t=A().getNumber(`WEBGL_VERSION`);if(e==null?this.gl=UN(t):(this.gl=e,HN(t,e)),e=this.gl,A().getNumber(`WEBGL_VERSION`)===2){let t=e;this.createVertexArray=()=>Y(t,()=>t.createVertexArray()),this.bindVertexArray=e=>Y(t,()=>t.bindVertexArray(e)),this.deleteVertexArray=e=>Y(t,()=>t.deleteVertexArray(e)),this.getVertexArray=()=>Y(t,()=>t.getParameter(t.VERTEX_ARRAY_BINDING))}else if(e!=null){let t=e.getExtension(`OES_vertex_array_object`);if(t==null)throw Error(`All WebGL1 implementations are expected to offer OES_vertex_array_object.`);this.createVertexArray=()=>Y(e,()=>t.createVertexArrayOES()),this.bindVertexArray=n=>Y(e,()=>t.bindVertexArrayOES(n)),this.deleteVertexArray=n=>Y(e,()=>t.deleteVertexArrayOES(n)),this.getVertexArray=()=>Y(e,()=>e.getParameter(t.VERTEX_ARRAY_BINDING_OES))}let n=`WEBGL_color_buffer_float`,r=`EXT_color_buffer_half_float`;if(this.parallelCompilationExtension=this.gl.getExtension(`KHR_parallel_shader_compile`),A().getNumber(`WEBGL_VERSION`)===1){let e=`OES_texture_half_float`;if(this.textureFloatExtension=oP(this.gl,`OES_texture_float`),VP(this.gl,e))this.textureHalfFloatExtension=oP(this.gl,e);else if(A().get(`WEBGL_FORCE_F16_TEXTURES`))throw Error(`GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.`);if(this.colorBufferFloatExtension=this.gl.getExtension(n),VP(this.gl,r))this.colorBufferHalfFloatExtension=oP(this.gl,r);else if(A().get(`WEBGL_FORCE_F16_TEXTURES`))throw Error(`GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.`)}else if(n=`EXT_color_buffer_float`,VP(this.gl,n))this.colorBufferFloatExtension=this.gl.getExtension(n);else if(VP(this.gl,r))this.colorBufferHalfFloatExtension=this.gl.getExtension(r);else throw Error(`GL context does not support color renderable floats`);this.vertexBuffer=lI(this.gl),this.indexBuffer=uI(this.gl),this.framebuffer=vP(this.gl),this.textureConfig=eP(this.gl,this.textureHalfFloatExtension)}get debug(){return A().getBool(`DEBUG`)}dispose(){if(this.disposed)return;this.program!=null&&console.warn(`Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing.`),this.outputTexture!=null&&console.warn(`Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.`);let e=this.gl;Y(e,()=>e.finish()),Y(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,null)),Y(e,()=>e.deleteFramebuffer(this.framebuffer)),Y(e,()=>e.bindBuffer(e.ARRAY_BUFFER,null)),Y(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,null)),Y(e,()=>e.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(e,t){return this.throwIfDisposed(),pI(this.gl,e,t,this.textureConfig)}createFloat16MatrixTexture(e,t){return this.throwIfDisposed(),hI(this.gl,e,t,this.textureConfig)}createUnsignedBytesMatrixTexture(e,t){return this.throwIfDisposed(),_I(this.gl,e,t,this.textureConfig)}uploadPixelDataToTexture(e,t){this.throwIfDisposed(),wI(this.gl,e,t)}uploadDenseMatrixToTexture(e,t,n,r){this.throwIfDisposed(),CI(this.gl,e,t,n,r,this.textureConfig)}createFloat16PackedMatrixTexture(e,t){return this.throwIfDisposed(),xI(this.gl,e,t,this.textureConfig)}createPackedMatrixTexture(e,t){return this.throwIfDisposed(),yI(this.gl,e,t,this.textureConfig)}deleteMatrixTexture(e){this.throwIfDisposed(),this.outputTexture===e&&(TP(this.gl,this.framebuffer),this.outputTexture=null),Y(this.gl,()=>this.gl.deleteTexture(e))}downloadByteEncodedFloatMatrixFromOutputTexture(e,t,n){return this.downloadMatrixDriver(e,()=>DI(this.gl,t,n,this.textureConfig))}downloadPackedMatrixFromBuffer(e,t,n,r,i,a){return OI(this.gl,e,t,n,r,i,a,this.textureConfig)}downloadFloat32MatrixFromBuffer(e,t){return EI(this.gl,e,t)}createBufferFromTexture(e,t,n){this.bindTextureToFrameBuffer(e);let r=TI(this.gl,t,n,this.textureConfig);return this.unbindTextureToFrameBuffer(),r}createAndWaitForFence(){let e=this.createFence(this.gl);return this.pollFence(e)}createFence(e){let t,n;if(A().getBool(`WEBGL_FENCE_API_ENABLED`)){let r=e,i=r.fenceSync(r.SYNC_GPU_COMMANDS_COMPLETE,0);e.flush(),n=()=>{let e=r.clientWaitSync(i,0,0);return e===r.ALREADY_SIGNALED||e===r.CONDITION_SATISFIED},t=i}else A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)>0?(t=this.beginQuery(),this.endQuery(),n=()=>this.isQueryAvailable(t,A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))):n=()=>!0;return{query:t,isFencePassed:n}}downloadMatrixFromPackedTexture(e,t,n){return this.downloadMatrixDriver(e,()=>kI(this.gl,t,n))}createProgram(e){this.throwIfDisposed();let t=this.gl;this.vertexShader??=cI(t);let n=dP(t);Y(t,()=>t.attachShader(n,this.vertexShader)),Y(t,()=>t.attachShader(n,e)),fP(t,n);let r=Object.assign(n,{vao:this.createVertexArray()});return this.debug&&pP(t,r),r}buildVao(e){this.setProgram(e),this.bindVertexArray(e.vao);let t=this.gl;Y(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,this.indexBuffer)),SI(t,e,this.vertexBuffer)}deleteProgram(e){this.throwIfDisposed(),e===this.program&&(this.program=null),e!=null&&(Y(this.gl,()=>this.gl.deleteProgram(e)),this.deleteVertexArray(e.vao))}setProgram(e){this.throwIfDisposed(),this.program=e,this.program!=null&&this.debug&&pP(this.gl,this.program),Y(this.gl,()=>this.gl.useProgram(e))}getUniformLocation(e,t,n=!0){return this.throwIfDisposed(),n?xP(this.gl,e,t):SP(this.gl,e,t)}getAttributeLocation(e,t){return this.throwIfDisposed(),Y(this.gl,()=>this.gl.getAttribLocation(e,t))}getUniformLocationNoThrow(e,t){return this.throwIfDisposed(),this.gl.getUniformLocation(e,t)}setInputMatrixTexture(e,t,n){this.throwIfDisposed(),this.throwIfNoProgram(),CP(this.gl,e,t,n)}setOutputMatrixTexture(e,t,n){this.setOutputMatrixTextureDriver(e,n,t)}setOutputPackedMatrixTexture(e,t,n){this.throwIfDisposed();let[r,i]=QN(t,n);this.setOutputMatrixTextureDriver(e,r,i)}setOutputMatrixWriteRegion(e,t,n,r){this.setOutputMatrixWriteRegionDriver(n,e,r,t)}setOutputPackedMatrixWriteRegion(e,t,n,r){throw Error(`setOutputPackedMatrixWriteRegion not implemented.`)}debugValidate(){this.program!=null&&pP(this.gl,this.program),EP(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();let e=this.gl;if(this.debug){let e=this.getVertexArray();console.assert(e===this.program.vao,`VAO changed between setProgram and executeProgram!`),this.debugValidate()}Y(e,()=>e.drawElements(e.TRIANGLES,6,e.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),Y(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return this.disjointQueryTimerExtension??=oP(this.gl,A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2?`EXT_disjoint_timer_query_webgl2`:`EXT_disjoint_timer_query`),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2){let e=this.gl,t=this.getQueryTimerExtensionWebGL2(),n=e.createQuery();return e.beginQuery(t.TIME_ELAPSED_EXT,n),n}let e=this.getQueryTimerExtensionWebGL1(),t=e.createQueryEXT();return e.beginQueryEXT(e.TIME_ELAPSED_EXT,t),t}endQuery(){if(A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`)===2){let e=this.gl,t=this.getQueryTimerExtensionWebGL2();e.endQuery(t.TIME_ELAPSED_EXT);return}let e=this.getQueryTimerExtensionWebGL1();e.endQueryEXT(e.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(e){return await ee(()=>this.disposed||this.isQueryAvailable(e,A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))),this.getQueryTime(e,A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION`))}getQueryTime(e,t){if(t===0)return null;if(t===2){let t=this.gl;return t.getQueryParameter(e,t.QUERY_RESULT)/1e6}{let t=this.getQueryTimerExtensionWebGL1();return t.getQueryObjectEXT(e,t.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(e,t){if(t===0)return!0;if(t===2){let t=this.gl,n=this.getQueryTimerExtensionWebGL2(),r=t.getQueryParameter(e,t.QUERY_RESULT_AVAILABLE);return this.disjoint??=this.gl.getParameter(n.GPU_DISJOINT_EXT),r&&!this.disjoint}{let t=this.getQueryTimerExtensionWebGL1(),n=t.getQueryObjectEXT(e,t.QUERY_RESULT_AVAILABLE_EXT);return this.disjoint??=this.gl.getParameter(t.GPU_DISJOINT_EXT),n&&!this.disjoint}}pollFence(e){return new Promise(t=>{this.addItemToPoll(()=>e.isFencePassed(),()=>t())})}pollItems(){let e=jI(this.itemsToPoll.map(e=>e.isDoneFn));for(let t=0;t<=e;++t){let{resolveFn:e}=this.itemsToPoll[t];e()}this.itemsToPoll=this.itemsToPoll.slice(e+1)}addItemToPoll(e,t){if(this.itemsToPoll.push({isDoneFn:e,resolveFn:t}),this.itemsToPoll.length>1)return;let n;`setTimeoutCustom`in A().platform&&(n=A().platform.setTimeoutCustom.bind(A().platform)),ee(()=>(this.pollItems(),this.itemsToPoll.length===0),()=>0,null,n)}bindTextureToFrameBuffer(e){this.throwIfDisposed(),wP(this.gl,e,this.framebuffer),this.debug&&EP(this.gl)}unbindTextureToFrameBuffer(){this.outputTexture==null?TP(this.gl,this.framebuffer):(wP(this.gl,this.outputTexture,this.framebuffer),this.debug&&EP(this.gl))}downloadMatrixDriver(e,t){this.bindTextureToFrameBuffer(e);let n=t();return this.unbindTextureToFrameBuffer(),n}setOutputMatrixTextureDriver(e,t,n){this.throwIfDisposed();let r=this.gl;wP(r,e,this.framebuffer),this.debug&&EP(r),this.outputTexture=e,Y(r,()=>r.viewport(0,0,t,n)),Y(r,()=>r.scissor(0,0,t,n))}setOutputMatrixWriteRegionDriver(e,t,n,r){this.throwIfDisposed(),Y(this.gl,()=>this.gl.scissor(e,t,n,r))}throwIfDisposed(){if(this.disposed)throw Error(`Attempted to use disposed GPGPUContext.`)}throwIfNoProgram(){if(this.program==null)throw Error(`No GPU program is currently set.`)}};function jI(e){let t=0;for(;t<e.length&&e[t]();++t);return t-1}var{addImpl:MI,bincountImpl:NI,bincountReduceImpl:PI,bitwiseAndImpl:FI,castImpl:II,ceilImpl:LI,concatImpl:RI,equalImpl:zI,expImpl:BI,expm1Impl:VI,floorImpl:HI,gatherNdImpl:UI,gatherV2Impl:WI,greaterImpl:GI,greaterEqualImpl:KI,lessImpl:qI,lessEqualImpl:JI,linSpaceImpl:YI,logImpl:XI,maxImpl:ZI,maximumImpl:QI,minimumImpl:$I,multiplyImpl:eL,negImpl:tL,notEqualImpl:nL,prodImpl:rL,raggedGatherImpl:iL,raggedRangeImpl:aL,raggedTensorToTensorImpl:oL,rangeImpl:sL,rsqrtImpl:cL,scatterImpl:lL,sigmoidImpl:uL,simpleAbsImpl:dL,sliceImpl:fL,sparseFillEmptyRowsImpl:pL,sparseReshapeImpl:mL,sparseSegmentReductionImpl:hL,sqrtImpl:gL,staticRegexReplaceImpl:_L,stridedSliceImpl:vL,stringNGramsImpl:yL,stringSplitImpl:bL,stringToHashBucketFastImpl:xL,subImpl:SL,tileImpl:CL,topKImpl:wL,transposeImpl:TL,uniqueImpl:EL}=jO;function DL(e,t){return[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,t).map(t=>`${e}.${t}`)}function OL(e,t){return t===1?[e]:DL(e,t)}function kL(e,t){if(e===1)return`rc`;let n=``;for(let r=0;r<e;r++)n+=t[r],r<e-1&&(n+=`,`);return n}var AL=class{constructor(e){if(this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.enableShapeUniforms=eI(this.outputShape.length),this.rank===0)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{let e=OL(`rc`,this.rank),t=GF(this.rank),n=this.getOutOfBoundsCondition(e),r=this.getSetup(e),i=this.getOutput(e);this.userCode=`
        void main() {
          ${t} rc = getOutputCoords();

          if(${n}) {
            setOutput(vec4(0));
          } else {
            ${r}

            setOutput(vec4(${i}));
          }
        }
      `}}getSourceCoordsArr(e){let t=[];for(let n=0;n<=1;n++)for(let r=0;r<=1;r++){let i=`${n===0?`r`:`rp1`}, ${r===0?`c`:`cp1`}`;for(let t=2;t<this.rank;t++)i=`${e[e.length-1-t]},`+i;t.push(i)}return t}getOutOfBoundsCondition(e){if(this.rank===1)return`rc > ${this.enableShapeUniforms?`outShape`:this.outputShape[0]}`;let t=``;for(let n=this.rank-2;n<this.rank;n++)t+=`${e[n]} >= ${this.enableShapeUniforms?`outShape[${n}]`:this.outputShape[n]}`,n<this.rank-1&&(t+=`||`);return t}getSetup(e){if(this.rank===1)return``;let t=e.slice(-2),n=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],r=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${t[0]};
      int c = ${t[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${n};
      bool rEdge = rp1 >= ${r};
    `}getOutput(e){let t=this.getSourceCoordsArr(e);return this.rank===1?`getA(rc), (rc + 1 >= ${this.enableShapeUniforms?`outShape`:this.outputShape[0]} ? 0. : getA(rc + 1)), 0, 0`:`getA(${t[0]}),
            cEdge ? 0. : getA(${t[1]}),
            rEdge ? 0. : getA(${t[2]}),
            rEdge || cEdge ? 0. : getA(${t[3]})`}},jL=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`inputShape`,type:`ivec3`}],this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length);let n=``;for(let e=0;e<4;e++){let t=`thisRC = rc;`;e%2==1&&(t+=`thisRC.z += 1;`),e>1&&(t+=`thisRC.y += 1;`),n+=`
        ${t}
        ${e>0?`if(thisRC.y < rows && thisRC.z < cols){`:``}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${e}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${e>0?`}`:``}
      `}this.userCode=`
      ${ML(t,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?tF():eF(e)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?`outShape[1]`:e[1]};
        int cols = ${this.enableShapeUniforms?`outShape[2]`:e[2]};

        ${n}

        setOutput(result);
      }
    `}};function ML(e,t){return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${t?$P([`r`,`c`,`d`],`inputShape`):XP([`r`,`c`,`d`],e)}
      return ivec3(r, c, d);
    }
  `}var NL=class{constructor(e){this.gpgpu=e,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.usedTextures={},this.logEnabled=!1}acquireTexture(e,t,n){let r=RL(t,n),i=zL(e,r,n);i in this.freeTextures||(this.freeTextures[i]=[]),i in this.usedTextures||(this.usedTextures[i]=[]);let a=FL(e,r,this.gpgpu.gl,this.gpgpu.textureConfig,n);if(this.freeTextures[i].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=a,this.log();let e=this.freeTextures[i].pop();return this.usedTextures[i].push(e),e}let o;return r===JN.PACKED_2X2_FLOAT32?o=this.gpgpu.createPackedMatrixTexture(e[0],e[1]):r===JN.PACKED_2X2_FLOAT16?o=this.gpgpu.createFloat16PackedMatrixTexture(e[0],e[1]):r===JN.UNPACKED_FLOAT32?o=this.gpgpu.createFloat32MatrixTexture(e[0],e[1]):r===JN.UNPACKED_FLOAT16?o=this.gpgpu.createFloat16MatrixTexture(e[0],e[1]):r===JN.PACKED_4X1_UNSIGNED_BYTE&&(o=this.gpgpu.createUnsignedBytesMatrixTexture(e[0],e[1])),this.usedTextures[i].push(o),this.numUsedTextures++,this._numBytesAllocated+=a,this.log(),o}releaseTexture(e,t,n,r){if(this.freeTextures==null)return;let i=RL(n,r),a=zL(t,i,r);a in this.freeTextures||(this.freeTextures[a]=[]);let o=FL(t,i,this.gpgpu.gl,this.gpgpu.textureConfig,r),s=A().getNumber(`WEBGL_DELETE_TEXTURE_THRESHOLD`);s!==-1&&this._numBytesAllocated>s?(this.gpgpu.deleteMatrixTexture(e.texture),this._numBytesAllocated-=o):(this.freeTextures[a].push(e),this.numFreeTextures++,this._numBytesFree+=o),this.numUsedTextures--;let c=this.usedTextures[a],l=c&&c.indexOf(e);if(l==null||l<0)throw Error(`Cannot release a texture that was never provided by this texture manager`);c[l]=c[c.length-1],c.pop(),this.log()}log(){if(!this.logEnabled)return;let e=this.numFreeTextures+this.numUsedTextures;console.log(`Free/Used`,`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${e})`);let t=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*t)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(this.freeTextures!=null){for(let e in this.freeTextures)this.freeTextures[e].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});for(let e in this.usedTextures)this.usedTextures[e].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}};function PL(e,t){let n=e;if(t===n.R32F)return 4;if(t===n.R16F)return 2;if(t===n.RGBA32F||t===e.RGBA)return 16;if(t===n.RGBA16F)return 8;if(t===n.RGBA8)return 4;throw Error(`Unknown internal format ${t}`)}function FL(e,t,n,r,i){let a=IL(t,r),o;if(i){let[t,n]=QN(e[0],e[1]);o=t*n}else{let[t,n]=YN(e[0],e[1]);o=t*n}let s=PL(n,a);return o*s}function IL(e,t){switch(e){case JN.PACKED_2X2_FLOAT32:return vI(t);case JN.PACKED_2X2_FLOAT16:return bI(t);case JN.UNPACKED_FLOAT32:return fI(t);case JN.UNPACKED_FLOAT16:return mI(t);case JN.PACKED_4X1_UNSIGNED_BYTE:return gI(t);default:throw Error(`Unknown physical texture type ${e}`)}}function LL(e){return A().getBool(`WEBGL_RENDER_FLOAT32_ENABLED`)?e?JN.PACKED_2X2_FLOAT32:JN.UNPACKED_FLOAT32:e?JN.PACKED_2X2_FLOAT16:JN.UNPACKED_FLOAT16}function RL(e,t){if(e===qN.UPLOAD)return JN.PACKED_2X2_FLOAT32;if(e===qN.RENDER||e==null)return LL(t);if(e===qN.DOWNLOAD||e===qN.PIXELS)return JN.PACKED_4X1_UNSIGNED_BYTE;throw Error(`Unknown logical texture type ${e}`)}function zL(e,t,n){return`${e[0]}_${e[1]}_${t}_${n}`}var BL=class{constructor(e,t){this.variableNames=[`A`],this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${t}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}},VL=`if (isnan(x)) return x;`,HL=`return x;`,UL=`return abs(x);`,WL=`return (x >= 0.0) ? x : (exp(x) - 1.0);`,GL=VL+`
  return (x < 0.0) ? 0.0 : x;
`,KL=VL+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,qL=`return x;`,JL=`return 1.0 / (1.0 + exp(-1.0 * x));`,YL=`return x;`,XL=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,ZL=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,QL=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,$L=`return 1.0 / (1.0 + exp(-1.0 * x));`,eR=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${t}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}},tR=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length);let t=e.length,n=OL(`rc`,t),r=GF(t),i=kL(t,n),a=n.slice(-2),o=t<=1?`rc`:`vec2(${a.join(`,`)})`;this.userCode=`
      void main() {
        ${r} rc = getOutputCoords();
        vec4 packedInput = getA(${i});

        setOutput(getChannel(packedInput, ${o}));
      }
    `}},nR=qf,rR=1e-7,iR=1e-4,aR={};function oR(e){return e in aR||(aR[e]={}),aR[e]}var sR=A().getNumber(`CPU_HANDOFF_SIZE_THRESHOLD`),cR=600;function lR(){return A().global.screen==null?1024:A().global.screen.height*A().global.screen.width*window.devicePixelRatio*cR/1024/1024}var uR=class e extends p{nextDataId(){return e.nextDataId++}constructor(e){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!A().getBool(`HAS_WEBGL`))throw Error(`WebGL is not supported on this device`);let t;e==null?(t=new AI(UN(A().getNumber(`WEBGL_VERSION`))),this.binaryCache=oR(A().getNumber(`WEBGL_VERSION`)),this.gpgpuCreatedLocally=!0):(t=e instanceof AI?e:new AI(UN(A().getNumber(`WEBGL_VERSION`),e)),this.binaryCache={},this.gpgpuCreatedLocally=!1),this.gpgpu=t,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new NL(this.gpgpu),this.numMBBeforeWarning=lR(),this.texData=new f(this,_a())}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}writeTexture(e,t,n,r,i,a){let o=this.makeTensorInfo(t,n),s=this.texData.get(o.dataId);s.isPacked=!1,s.texture={texture:e,texShape:[r,i]},s.texShape=[r,i];let c=new oI(MP(t),!1,a),l=this.runWebGLProgram(c,[o],n,[[r,i]]);return l.shape=t,s.texture=null,this.disposeIntermediateTensorInfo(o),l.dataId}write(e,t,n){if((A().getBool(`WEBGL_CHECK_NUMERICAL_PROBLEMS`)||A().getBool(`DEBUG`))&&this.checkNumericalProblems(e),n===`complex64`&&e!=null)throw Error(`Cannot write to a complex64 dtype. Please use tf.complex(real, imag).`);let r={id:this.nextDataId()};return this.texData.set(r,{shape:t,dtype:n,values:e,usage:qN.UPLOAD,refCount:1}),r}refCount(e){return this.texData.has(e)?this.texData.get(e).refCount:0}incRef(e){let t=this.texData.get(e);t.refCount++}decRef(e){if(this.texData.has(e)){let t=this.texData.get(e);t.refCount--}}move(e,t,n,r,i){if(A().getBool(`DEBUG`)&&this.checkNumericalProblems(t),r===`complex64`)throw Error(`Cannot write to a complex64 dtype. Please use tf.complex(real, imag).`);this.texData.set(e,{shape:n,dtype:r,values:t,usage:qN.UPLOAD,refCount:i})}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}readSync(e){let{values:t,dtype:n,complexTensorInfos:r,slice:i,shape:a,isPacked:o}=this.texData.get(e);if(i!=null){let t;t=o?new eR(a,qL):new BL(a,qL);let r=this.runWebGLProgram(t,[{dataId:e,shape:a,dtype:n}],n),i=this.readSync(r.dataId);return this.disposeIntermediateTensorInfo(r),i}if(t!=null)return this.convertAndCacheOnCPU(e);if(n===`string`)return t;let s=this.activeTimers!=null,c;s&&(c=li());let l;return l=n===`complex64`?Bh(this.readSync(r.real.dataId),this.readSync(r.imag.dataId)):this.getValuesFromTexture(e),s&&(this.downloadWaitMs+=li()-c),this.convertAndCacheOnCPU(e,l)}async read(e){if(this.pendingRead.has(e)){let t=this.pendingRead.get(e);return new Promise(e=>t.push(e))}let{values:t,shape:n,slice:r,dtype:i,complexTensorInfos:a,isPacked:o}=this.texData.get(e);if(r!=null){let t;t=o?new eR(n,qL):new BL(n,qL);let r=this.runWebGLProgram(t,[{dataId:e,shape:n,dtype:i}],i),a=this.read(r.dataId);return this.disposeIntermediateTensorInfo(r),a}if(t!=null)return this.convertAndCacheOnCPU(e);if(A().getBool(`DEBUG`)&&!A().getBool(`WEBGL_DOWNLOAD_FLOAT_ENABLED`)&&A().getNumber(`WEBGL_VERSION`)===2)throw Error(`tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.`);let s=null,c;if(i!==`complex64`&&A().get(`WEBGL_BUFFER_SUPPORTED`)){c=this.decode(e);let t=this.texData.get(c.dataId);s=this.gpgpu.createBufferFromTexture(t.texture.texture,...ZN(n))}this.pendingRead.set(e,[]),i!==`complex64`&&await this.gpgpu.createAndWaitForFence();let l;if(i===`complex64`){let e=await Promise.all([this.read(a.real.dataId),this.read(a.imag.dataId)]),t=e[0],n=e[1];l=Bh(t,n)}else if(s==null)l=this.getValuesFromTexture(e);else{let e=C(n);l=this.gpgpu.downloadFloat32MatrixFromBuffer(s,e)}if(c!=null&&this.disposeIntermediateTensorInfo(c),s!=null){let e=this.gpgpu.gl;Y(e,()=>e.deleteBuffer(s))}let u=this.convertAndCacheOnCPU(e,l),d=this.pendingRead.get(e);return this.pendingRead.delete(e),d.forEach(e=>e(u)),this.pendingDisposal.has(e)&&(this.pendingDisposal.delete(e),this.disposeData(e)&&_a().removeDataId(e,this),this.pendingDeletes--),u}readToGPU(e,t={}){let{values:n,shape:r,slice:i,dtype:a,isPacked:o,texture:s}=this.texData.get(e);if(a===`complex64`)throw Error(`Does not support reading texture for complex64 dtype.`);if(i!=null){let n;n=o?new eR(r,qL):new BL(r,qL);let i=this.runWebGLProgram(n,[{dataId:e,shape:r,dtype:a}],a),s=this.readToGPU(i,t);return this.disposeIntermediateTensorInfo(i),s}if(s==null)throw Error(n==null?`There is no data on GPU or CPU.`:`Data is not on GPU but on CPU.`);let c=this.decode(e,t.customTexShape),l=_a().makeTensorFromTensorInfo(c),u=this.texData.get(c.dataId);return Object.assign({tensorRef:l},u.texture)}bufferSync(e){let t=this.readSync(e.dataId);if(e.dtype===`string`)try{let n=t.map(e=>di(e));return _o(e.shape,e.dtype,n)}catch{throw Error(`Failed to decode encoded string bytes into utf-8`)}return _o(e.shape,e.dtype,t)}checkNumericalProblems(e){if(e!=null)for(let t=0;t<e.length;t++){let n=e[t];if(!iP(n))throw A().getBool(`WEBGL_RENDER_FLOAT32_CAPABLE`)?Error(`The value ${n} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`):Error(`The value ${n} cannot be represented on this device.`)}}getValuesFromTexture(e){let{shape:t,dtype:n,isPacked:r}=this.texData.get(e),i=C(t);if(A().getBool(`WEBGL_DOWNLOAD_FLOAT_ENABLED`)){let n=this.decode(e),r=this.texData.get(n.dataId),a=this.gpgpu.downloadMatrixFromPackedTexture(r.texture.texture,...ZN(t)).subarray(0,i);return this.disposeIntermediateTensorInfo(n),a}let a=A().getBool(`WEBGL_PACK`)&&r===!0,o=a?MP(t):t,s=a?new iI(o):new rI(o),c=this.runWebGLProgram(s,[{shape:o,dtype:n,dataId:e}],`float32`),l=this.texData.get(c.dataId),u=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(l.texture.texture,l.texShape[0],l.texShape[1]).subarray(0,i);return this.disposeIntermediateTensorInfo(c),u}timerAvailable(){return A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0}time(e){let t=this.activeTimers,n=[],r=!1;this.programTimersStack==null?(this.programTimersStack=n,r=!0):this.activeTimers.push(n),this.activeTimers=n,e();let i=pi(this.activeTimers.map(e=>e.query)).filter(e=>e!=null),a=pi(this.activeTimers.map(e=>e.name)).filter(e=>e!=null);this.activeTimers=t,r&&(this.programTimersStack=null);let o={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if(A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0){let e=await Promise.all(i);o.kernelMs=y(e),o.getExtraProfileInfo=()=>e.map((e,t)=>({name:a[t],ms:e})).map(e=>`${e.name}: ${e.ms}`).join(`, `)}else o.kernelMs={error:`WebGL query timers are not supported in this environment.`};return this.uploadWaitMs=0,this.downloadWaitMs=0,o})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0?this.gpgpu.beginQuery():{startMs:li(),endMs:null}}endTimer(e){return A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0?(this.gpgpu.endQuery(),e):(e.endMs=li(),e)}async getQueryTime(e){if(A().getNumber(`WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE`)>0)return this.gpgpu.waitForQueryAndGetTime(e);let t=e;return t.endMs-t.startMs}disposeData(e,t=!1){if(this.pendingDisposal.has(e))return!1;if(!this.texData.has(e))return!0;if(t?this.texData.get(e).refCount=0:this.texData.get(e).refCount--,!t&&this.texData.get(e).refCount>0)return!1;if(this.pendingRead.has(e))return this.pendingDisposal.add(e),this.pendingDeletes++,!1;this.releaseGPUData(e);let{complexTensorInfos:n}=this.texData.get(e);return n!=null&&(this.disposeData(n.real.dataId,t),this.disposeData(n.imag.dataId,t)),this.texData.delete(e),!0}releaseGPUData(e){let{texture:t,dtype:n,texShape:r,usage:i,isPacked:a,slice:o}=this.texData.get(e),s=o&&o.origDataId||e,c=this.dataRefCount.get(s);c>1?this.dataRefCount.set(s,c-1):(this.dataRefCount.delete(s),t!=null&&(this.numBytesInGPU-=this.computeBytes(r,n),this.textureManager.releaseTexture(t,r,i,a)));let l=this.texData.get(e);l.texture=null,l.texShape=null,l.isPacked=!1,l.slice=null}getTexture(e){return this.uploadToGPU(e),this.texData.get(e).texture.texture}getDataInfo(e){return this.texData.get(e)}shouldExecuteOnCPU(e,t=sR){return A().getBool(`WEBGL_CPU_FORWARD`)&&e.every(e=>this.texData.get(e.dataId).texture==null&&C(e.shape)<t)}getGPGPUContext(){return this.gpgpu}where(e){jr(`tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead`);let t=e.dataSync();return nR(e.shape,t)}packedUnaryOp(e,t,n){let r=new eR(e.shape,t),i=this.compileAndRun(r,[e],n);return _a().makeTensorFromTensorInfo(i)}abs(e){if(this.shouldExecuteOnCPU([e])&&e.dtype!==`complex64`){let t=dL(this.texData.get(e.dataId).values);return this.makeOutput(e.shape,e.dtype,t)}if(A().getBool(`WEBGL_PACK_UNARY_OPERATIONS`))return this.packedUnaryOp(e,UL,e.dtype);let t=new BL(e.shape,UL),n=this.compileAndRun(t,[e]);return _a().makeTensorFromTensorInfo(n)}makeTensorInfo(e,t,n){let r;if(t===`string`&&n!=null&&n.length>0&&ue(n[0])){let i=n.map(e=>ui(e));r=this.write(i,e,t)}else r=this.write(n,e,t);return this.texData.get(r).usage=null,{dataId:r,shape:e,dtype:t}}makeOutput(e,t,n){return _a().makeTensorFromTensorInfo(this.makeTensorInfo(e,t,n),this)}unpackTensor(e){let t=new tR(e.shape);return this.runWebGLProgram(t,[e],e.dtype)}packTensor(e){let t=new AL(e.shape);return this.runWebGLProgram(t,[e],e.dtype,null,!0)}packedReshape(e,t){let n=[AP(e.shape),...jP(e.shape)],r={dtype:e.dtype,shape:n,dataId:e.dataId},i=new jL([AP(t),...jP(t)],n),a=[n],o=this.runWebGLProgram(i,[r],e.dtype,a,!0);return{dataId:o.dataId,shape:t,dtype:o.dtype}}decode(e,t){let{isPacked:n,shape:r,dtype:i}=this.texData.get(e);t!=null&&b(C(r)<=t[0]*t[1]*4,()=>`customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.`);let a=MP(r),o;o=n?new nI(a):new tI(a);let s=[t??ZN(a)];return{dtype:i,shape:r,dataId:this.runWebGLProgram(o,[{shape:a,dtype:i,dataId:e}],i,s,!0,t).dataId}}runWebGLProgram(e,t,n,r,i=!1,a){let o=this.makeTensorInfo(e.outputShape,n),s=this.texData.get(o.dataId);if(e.packedOutput&&(s.isPacked=!0),e.outPackingScheme===KN.DENSE&&(s.texShape=(a??ZN(e.outputShape)).map(e=>e*2)),e.outTexUsage!=null&&(s.usage=e.outTexUsage),C(o.shape)===0)return s.values=re(o.dtype,0),o;let c=[],l=t.map(t=>{if(t.dtype===`complex64`)throw Error(`GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.`);let n=this.texData.get(t.dataId);if(n.texture==null){if(!e.packedInputs&&C(t.shape)<=A().getNumber(`WEBGL_SIZE_UPLOAD_UNIFORM`))return{shape:t.shape,texData:null,isUniform:!0,uniformValues:n.values};e.packedInputs&&(n.isPacked=!0,n.shape=t.shape)}if(this.uploadToGPU(t.dataId),!!n.isPacked!=!!e.packedInputs)t=n.isPacked?this.unpackTensor(t):this.packTensor(t),c.push(t),n=this.texData.get(t.dataId);else if(n.isPacked&&!FP(n.shape,t.shape)){let e=t,r=t.shape;t.shape=n.shape,t=this.packedReshape(t,r),c.push(t),n=this.texData.get(t.dataId),e.shape=r}return{shape:t.shape,texData:n,isUniform:!1}});this.uploadToGPU(o.dataId);let u={shape:o.shape,texData:s,isUniform:!1},d=$F(e,l,u),f=this.getAndSaveBinary(d,()=>YF(this.gpgpu,e,l,u)),p=this.activeTimers!=null,m;p&&(m=this.startTimer()),A().get(`ENGINE_COMPILE_ONLY`)||QF(this.gpgpu,f,l,u,r),c.forEach(e=>this.disposeIntermediateTensorInfo(e)),p&&(m=this.endTimer(m),this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime(m)}));let h=A().getNumber(`WEBGL_FLUSH_THRESHOLD`);if(h>0){let e=li();e-this.lastGlFlushTime>h&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=e)}if(!A().getBool(`WEBGL_LAZILY_UNPACK`)&&s.isPacked&&i===!1){let e=this.unpackTensor(o);return this.disposeIntermediateTensorInfo(o),e}return o}compileAndRun(e,t,n,r,i=!1){return n||=t[0].dtype,this.runWebGLProgram(e,t,n,r,i)}getAndSaveBinary(e,t){return e in this.binaryCache||(this.binaryCache[e]=t()),this.binaryCache[e]}getTextureManager(){return this.textureManager}dispose(){this.disposed||=(A().getBool(`IS_TEST`)||Object.keys(this.binaryCache).forEach(e=>{this.gpgpu.deleteProgram(this.binaryCache[e].webGLProgram),delete this.binaryCache[e]}),this.textureManager.dispose(),this.canvas!=null&&typeof HTMLCanvasElement<`u`&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),!0)}floatPrecision(){return this.floatPrecisionValue??=F(()=>{if(!A().get(`WEBGL_RENDER_FLOAT32_ENABLED`)){let e=A().getBool(`DEBUG`);A().set(`DEBUG`,!1);let t=this.abs(H(1e-8)).dataSync()[0];if(A().set(`DEBUG`,e),t>0)return 32}return 16}),this.floatPrecisionValue}epsilon(){return this.floatPrecision()===32?rR:iR}uploadToGPU(e){let t=this.texData.get(e),{shape:n,dtype:r,values:i,texture:a,usage:o,isPacked:s}=t;if(a!=null)return;let c=this.activeTimers!=null,l;c&&(l=li());let u=t.texShape;if(u??(u=NP(n,s),t.texShape=u),i!=null){let e=MP(n),a,o=u[1],d=u[0],f=i instanceof Uint8Array||i instanceof Uint8ClampedArray;(s||!f)&&([o,d]=QN(u[0],u[1])),a=s?new sI(e,f):new oI(e,f);let p=f?[d,o]:u,m=this.makeTensorInfo(p,r),h=this.texData.get(m.dataId);h.usage=f?qN.PIXELS:qN.UPLOAD,h.texShape=p,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(m.dataId),o,d,i);let g=[[d,o]],_=this.runWebGLProgram(a,[m],r,g,!0),v=this.texData.get(_.dataId);t.texShape=v.texShape,t.isPacked=v.isPacked,t.usage=v.usage,A().get(`ENGINE_COMPILE_ONLY`)?this.disposeData(_.dataId):(t.texture=v.texture,t.values=null,this.texData.delete(_.dataId)),this.disposeIntermediateTensorInfo(m),c&&(this.uploadWaitMs+=li()-l)}else t.texture=this.acquireTexture(u,o,r,s)}convertAndCacheOnCPU(e,t){let n=this.texData.get(e),{dtype:r}=n;return t!=null&&(n.values=dR(t,r)),n.values}acquireTexture(e,t,n,r){if(this.numBytesInGPU+=this.computeBytes(e,n),!this.warnedAboutMemory&&this.numBytesInGPU>this.numMBBeforeWarning*1024*1024){let e=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${e} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(e,t,r)}computeBytes(e,t){return e[0]*e[1]*ce(t)}checkCompileCompletion(){for(let[,e]of Object.entries(this.binaryCache))this.checkCompletion_(e)}async checkCompileCompletionAsync(){let e=[];if(this.gpgpu.parallelCompilationExtension){for(let[,t]of Object.entries(this.binaryCache))e.push(this.checkCompletionAsync_(t));return Promise.all(e)}for(let[,t]of Object.entries(this.binaryCache)){let n=new Promise(e=>{try{this.checkCompletion_(t),e(!0)}catch(e){throw e}});e.push(n)}return Promise.all(e)}async checkCompletionAsync_(e){return this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(e):(await _h(),this.checkCompletionAsync_(e))}checkCompletion_(e){if(this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.gl.LINK_STATUS)===!1)throw console.log(this.gpgpu.gl.getProgramInfoLog(e.webGLProgram)),this.gpgpu.gl.getShaderParameter(e.fragmentShader,this.gpgpu.gl.COMPILE_STATUS)===!1?(uP(e.source,this.gpgpu.gl.getShaderInfoLog(e.fragmentShader)),Error(`Failed to compile fragment shader.`)):Error(`Failed to link vertex and fragment shaders.`);return!0}getUniformLocations(){for(let e of Object.values(this.binaryCache)){this.gpgpu.buildVao(e.webGLProgram);let{variablesLocations:t,customUniformLocations:n,infLoc:r,nanLoc:i,outShapeLocation:a,outShapeStridesLocation:o,outTexShapeLocation:s}=XF(this.gpgpu,e.program,e.webGLProgram);e.variablesLocations=t,e.customUniformLocations=n,e.infLoc=r,e.nanLoc=i,e.outShapeLocation=a,e.outShapeStridesLocation=o,e.outTexShapeLocation=s}}createTensorFromGPUData(e,t,n){e.channels=e.channels||`RGBA`;let{texture:r,height:i,width:a,channels:o}=e,s=_a().backend;if(!s.gpgpu.gl.isTexture(r))throw Error(`The texture is invalid. Also, please make sure the texture and the TFJS WebGL backend are using the same canvas. If you want to use your own custom canvas, you have to create and use the custom TFJS WebGL backend created from the canvas through 'new tf.MathBackendWebGL(customCanvas)'.`);let c=s.writeTexture(r,t,n,i,a,o);return _a().makeTensorFromDataId(c,t,n,s)}};uR.nextDataId=0;function dR(e,t){if(t===`float32`||t===`complex64`)return e;if(t===`int32`||t===`bool`){let n=t===`int32`?new Int32Array(e.length):new Uint8Array(e.length);for(let t=0;t<n.length;++t)n[t]=Math.round(e[t]);return n}throw Error(`Unknown dtype ${t}`)}ia()&&wa(`webgl`,()=>new uR,2);var fR=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`,pR=class{constructor(e,t,n){this.variableNames=[`A`,`B`],this.outputShape=V(t,n),this.enableShapeUniforms=eI(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${e}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}},mR=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`,hR=class{constructor(e,t,n,r=!1){this.variableNames=[`A`,`B`],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=V(t,n);let i=this.outputShape.length;this.enableShapeUniforms=eI(i);let a=``;if(r){if(i===0||C(this.outputShape)===1)a=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else if(a=`
          ${GF(i)} coords = getOutputCoords();
        `,i===1)this.enableShapeUniforms?a+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:a+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{let e=OL(`coords`,i);this.enableShapeUniforms?a+=`
            bool nextRowOutOfBounds =
              (${e[i-2]} + 1) >= outShape[${i} - 2];
            bool nextColOutOfBounds =
              (${e[i-1]} + 1) >= outShape[${i} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:a+=`
            bool nextRowOutOfBounds =
              (${e[i-2]} + 1) >= ${this.outputShape[i-2]};
            bool nextColOutOfBounds =
              (${e[i-1]} + 1) >= ${this.outputShape[i-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${e}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${a}

        setOutput(result);
      }
    `}};function gR(e){let{inputs:t,backend:n}=e,{x:r}=t;return n.incRef(r.dataId),{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}var _R={kernelName:Wt,backendName:`webgl`,kernelFunc:gR};function vR(e){let{inputs:t,backend:n}=e,{real:r,imag:i}=t,a=n.makeTensorInfo(r.shape,`complex64`),o=n.texData.get(a.dataId);return o.complexTensorInfos={real:gR({inputs:{x:r},backend:n}),imag:gR({inputs:{x:i},backend:n})},a}var yR={kernelName:ot,backendName:`webgl`,kernelFunc:vR},bR=`return (a < 0.) ? b * a : a;`,xR=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function SR(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{alpha:a}=r,o=n.makeTensorInfo([],`float32`,oi(a,`float32`)),s=A().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new hR(xR,i.shape,o.shape):new pR(bR,i.shape,o.shape),c=n.runWebGLProgram(s,[i,o],`float32`);return n.disposeIntermediateTensorInfo(o),c}var CR={kernelName:Xt,backendName:`webgl`,kernelFunc:SR},wR=`return (a < 0.) ? b * a : a;`,TR=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function ER(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t,a=A().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new hR(TR,r.shape,i.shape):new pR(wR,r.shape,i.shape);return n.runWebGLProgram(a,[r,i],`float32`)}var DR={kernelName:En,backendName:`webgl`,kernelFunc:ER},OR=`if (isnan(x)) return x;`;function kR({opSnippet:e,packedOpSnippet:t,cpuKernelImpl:n,dtype:r}){return({inputs:i,backend:a})=>{let{x:o}=i,s=a,c=r||o.dtype;if(s.shouldExecuteOnCPU([o])&&n!=null){let e=n(s.texData.get(o.dataId).values,c);return s.makeTensorInfo(o.shape,c,e)}let l=A().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)&&t!=null,u;return u=l?new eR(o.shape,t):new BL(o.shape,e),s.runWebGLProgram(u,[o],c)}}function AR({opSnippet:e,packedOpSnippet:t,checkOutOfBounds:n=!1,supportsComplex:r=!1,cpuKernelImpl:i,dtype:a}){return({inputs:o,backend:s})=>{let{a:c,b:l}=o,u=s;if(r&&c.dtype===`complex64`){let t=u.texData.get(c.dataId),n=u.texData.get(l.dataId),[r,i]=[[t.complexTensorInfos.real,n.complexTensorInfos.real],[t.complexTensorInfos.imag,n.complexTensorInfos.imag]].map(t=>{let[n,r]=t,i={dataId:n.dataId,dtype:n.dtype,shape:c.shape},a={dataId:r.dataId,dtype:r.dtype,shape:l.shape},o=new pR(e,c.shape,l.shape);return u.runWebGLProgram(o,[i,a],Vi(n.dtype,r.dtype))}),a=vR({inputs:{real:r,imag:i},backend:u});return u.disposeIntermediateTensorInfo(r),u.disposeIntermediateTensorInfo(i),a}let d=a||Vi(c.dtype,l.dtype);if((c.dtype===`string`||l.dtype===`string`||u.shouldExecuteOnCPU([c,l]))&&i!=null){let e=u.texData.get(c.dataId).values,t=u.texData.get(l.dataId).values,n=c.dtype===`string`?Sg(e):e,r=c.dtype===`string`?Sg(t):t,[a,o]=i(c.shape,l.shape,n,r,d),s=u.makeTensorInfo(o,d),f=u.texData.get(s.dataId);return f.values=a,s}let f=A().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)&&t!=null,p;return p=f?new hR(t,c.shape,l.shape,n):new pR(e,c.shape,l.shape),u.runWebGLProgram(p,[c,l],d)}}function jR(e,t=!1){if(e===`linear`)return t?YL:HL;if(e===`relu`)return t?ZL:GL;if(e===`elu`)return t?XL:WL;if(e===`relu6`)return t?QL:KL;if(e===`prelu`)return t?TR:wR;if(e===`leakyrelu`)return t?xR:bR;if(e===`sigmoid`)return t?$L:JL;throw Error(`Activation ${e} has not been implemented for the WebGL backend.`)}var MR=class{constructor(e,t,n,r=!1,i=!1,a=!1,o=null,s=!1,c=!1){this.variableNames=[`matrixA`,`matrixB`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=n,this.enableShapeUniforms=eI(this.outputShape.length);let l=r?e[1]:e[2],u=Math.ceil(l/2),d=r?`i * 2, rc.y`:`rc.y, i * 2`,f=i?`rc.z, i * 2`:`i * 2, rc.z`,p=r?[`a.xxyy`,`a.zzww`]:[`a.xxzz`,`a.yyww`],m=i?[`b.xzxz`,`b.ywyw`]:[`b.xyxy`,`b.zwzw`],h=``,g=``;o&&(h=s?`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${o}
        }`:c?`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${o}
        }`:`vec4 activation(vec4 x) {
          ${o}
        }`,g=`result = activation(result);`);let _=a?`result += getBiasAtOutCoords();`:``;a&&this.variableNames.push(`bias`),s&&this.variableNames.push(`preluActivationWeights`),c&&this.variableNames.push(`leakyreluAlpha`);let v=`rc.x`,y=`rc.x`;e[0]<t[0]?v=`imod(rc.x, ${e[0]})`:t[0]<e[0]&&(y=`imod(rc.x, ${t[0]})`),this.userCode=`
      ${h}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${u}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${v};
        int batchB = ${y};
        for (int i = 0; i < ${u}; i++) {
          vec4 a = getMatrixA(batchA, ${d});
          vec4 b = getMatrixB(batchB, ${f});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${p[0]} * ${m[0]});
          result += (${p[1]} * ${m[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${_}

        ${g}

        setOutput(result);
      }
    `}},NR={REAL:`return areal * breal - aimag * bimag;`,IMAG:`return areal * bimag + aimag * breal;`},PR=class{constructor(e,t,n){this.variableNames=[`AReal`,`AImag`,`BReal`,`BImag`],this.outputShape=V(t,n),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${e}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}},FR=`return a * b;`;function IR(e){let{inputs:t,backend:n}=e,{a:r,b:i}=t,a=Vi(r.dtype,i.dtype);if(r.dtype===`complex64`){let e=n.texData.get(r.dataId),t=n.texData.get(i.dataId),a=new PR(NR.REAL,r.shape,i.shape),o=new PR(NR.IMAG,r.shape,i.shape),s=[{dataId:e.complexTensorInfos.real.dataId,dtype:e.complexTensorInfos.real.dtype,shape:r.shape},{dataId:e.complexTensorInfos.imag.dataId,dtype:e.complexTensorInfos.imag.dtype,shape:r.shape},{dataId:t.complexTensorInfos.real.dataId,dtype:t.complexTensorInfos.real.dtype,shape:i.shape},{dataId:t.complexTensorInfos.imag.dataId,dtype:t.complexTensorInfos.imag.dtype,shape:i.shape}],c=n.runWebGLProgram(a,s,`float32`),l=n.runWebGLProgram(o,s,`float32`),u=vR({inputs:{real:c,imag:l},backend:n});return n.disposeIntermediateTensorInfo(c),n.disposeIntermediateTensorInfo(l),u}if(n.shouldExecuteOnCPU([r,i])){let e=n.texData.get(r.dataId),t=n.texData.get(i.dataId),[o,s]=eL(r.shape,i.shape,e.values,t.values,a),c=n.makeTensorInfo(s,a),l=n.texData.get(c.dataId);return l.values=o,c}let o;return o=A().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new hR(FR,r.shape,i.shape):new pR(FR,r.shape,i.shape),n.runWebGLProgram(o,[r,i],a)}var LR={kernelName:_n,backendName:`webgl`,kernelFunc:IR};function RR(e,t,n){let r=[AP(e.shape),...jP(e.shape)],i={dtype:e.dtype,shape:r,dataId:e.dataId},a=new jL([AP(t),...jP(t)],r),o=[r],s=n.runWebGLProgram(a,[i],e.dtype,o,!0);return{dataId:s.dataId,shape:t,dtype:s.dtype}}function Z(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{shape:a}=r,o=n,s=C(i.shape),c=te(a,s),l=C(c);b(s===l,()=>`The new shape (${c}) has ${l} elements and the old shape (${i.shape}) has ${s} elements. The new shape and old shape must have the same number of elements.`);let u=o.texData.get(i.dataId);return u.isPacked&&!FP(i.shape,c)&&!(u.texture!==null&&FP(u.shape,c))?RR(i,c,o):(o.incRef(i.dataId),{dataId:i.dataId,shape:c,dtype:i.dtype})}var zR={kernelName:Fn,backendName:`webgl`,kernelFunc:Z},BR=class{constructor(e,t){this.variableNames=[`x`];let{windowSize:n,batchSize:r,inSize:i,outSize:a}=e;this.outputShape=[r,a];let o=Math.floor(n/4)*4,s=n%4,c=`sumValue += dot(values, ones);`;if(t!=null){let e=1/t;c=`sumValue += dot(values * ${T(e)?e.toPrecision(2):e}, ones);`}let l=``;i%n>0&&(l=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${l}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${n};

        float sumValue = 0.0;

        for (int i = 0; i < ${o}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${c}
        }

        int inIdx = inOffset + ${o};
        if (${s===1}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${c}
        } else if (${s===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${c}
        } else if (${s===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${c}
        }
        setOutput(sumValue);
      }
    `}},VR=class{constructor(e,t){this.variableNames=[`x`];let{windowSize:n,batchSize:r,inSize:i,outSize:a}=e;this.outputShape=[r,a];let o=`0.0`,s=``;t===`prod`?o=`1.0`:t===`min`?(o=`1.0 / 1e-20`,s=`min`):t===`max`&&(o=`-1.0 / 1e-20`,s=`max`);let c=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`sum`?c=`sumValue`:t===`prod`?c=`prodValue`:t===`all`?c=`allValue`:t===`any`&&(c=`anyValue`);let l=Math.floor(n/4)*4,u=n%4,d=`
      if (${t===`sum`}) {
        sumValue += dot(values, ones);
      } else if (${t===`prod`}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${s}(values, minMaxValue);
        if (${t===`min`} || ${t===`max`}) {
          minMaxValue = ${s}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,f=`vec4`;t===`all`?(o=`1.0`,d=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,f=`bvec4`):t===`any`&&(o=`0.0`,d=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,f=`bvec4`);let p=``;i%n>0&&(p=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${o};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${p}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${n};

        vec4 minMaxValue = vec4(${o});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${l}; i += 4) {
          int inIdx = inOffset + i;
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${d}
        }

        int inIdx = inOffset + ${l};
        if (${u===1}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${u===2}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${u===3}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${d}
        }
        setOutput(${c});
      }
    `}};function HR(e){let t=[];for(;t.length===0||t[t.length-1].outSize!==1;){let n=t.length?t[t.length-1].outSize:e[1],r=Th(n);t.push({inSize:n,windowSize:r,outSize:Math.ceil(n/r)})}return t}function UR(e,t,n,r){let i=HR(e.shape),a=e;for(let o=0;o<i.length;o++){let{inSize:s,windowSize:c,outSize:l}=i[o],u,d;u=n===`mean`?o===0?new BR({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l},s):new BR({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l}):new VR({windowSize:c,inSize:s,batchSize:e.shape[0],outSize:l},n),d=a,a=r.runWebGLProgram(u,[a],t),d.dataId!==e.dataId&&r.disposeIntermediateTensorInfo(d)}return a}var WR=class{constructor(e,t){this.variableNames=[`A`];let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.rank=n.length;let r=GF(this.rank),i=GR(t);this.userCode=`
    void main() {
      ${r} resRC = getOutputCoords();
      setOutput(getA(${i}));
    }
    `}};function GR(e){let t=e.length;if(t>6)throw Error(`Transpose for rank ${t} is not yet supported`);let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`,`resRC.u`,`resRC.v`],r=Array(t);for(let t=0;t<e.length;t++)r[e[t]]=n[t];return r.join()}var KR=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0;let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];if(this.outputShape=n,this.rank=n.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);let r=GF(this.rank),i=DL(`rc`,this.rank),a=Array(this.rank);for(let e=0;e<t.length;e++)a[t[e]]=i[e];let o=`vec2(${a.slice(-2).join()})`,s=`++${i[this.rank-1]} < ${n[this.rank-1]}`,c=`getChannel(getA(${a.join()}), ${o})`;this.userCode=`
    void main() {
      ${r} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${c};
      if(${s}) {
        result[1] = ${c};
      }
      --${i[this.rank-1]};
      if(++${i[this.rank-2]} < ${n[this.rank-2]}) {
        result[2] = ${c};
        if(${s}) {
          result[3] = ${c};
        }
      }
      setOutput(result);
    }
    `}};function qR(e,t,n){let r=A().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new KR(e.shape,t):new WR(e.shape,t);return n.runWebGLProgram(r,[e],e.dtype)}function JR(e,t,n,r){let i=t,a=e.shape.length,o=O(i,e.shape),s=o,c=ol(s,a),l=c!=null,u=e;l&&(u=qR(e,c,r),s=cl(s.length,a)),al(`sum`,s,a);let[d,f]=rl(u.shape,s),p=d;n&&(p=il(d,o));let m=C(f),h=C(e.shape)/m,g=Z({inputs:{x:u},attrs:{shape:[h,m]},backend:r}),_=UR(g,Hi(e.dtype),`sum`,r),v=Z({inputs:{x:_},attrs:{shape:p},backend:r});return r.disposeIntermediateTensorInfo(g),r.disposeIntermediateTensorInfo(_),l&&r.disposeIntermediateTensorInfo(u),v}function YR(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;return JR(i,a,o,n)}var XR={kernelName:`Sum`,backendName:`webgl`,kernelFunc:YR};function ZR(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{perm:a}=r,o=n,s=i.shape.length,c=Array(s);for(let e=0;e<c.length;e++)c[e]=i.shape[a[e]];let l;if(o.shouldExecuteOnCPU([i])){let e=o.texData.get(i.dataId).values,t=TL(e,i.shape,i.dtype,a,c);l=o.makeTensorInfo(c,i.dtype);let n=o.texData.get(l.dataId);n.values=t}else l=qR(i,a,o);return l}var QR={kernelName:br,backendName:`webgl`,kernelFunc:ZR};function $R({a:e,b:t,transposeA:n,transposeB:r,backend:i,bias:a=null,preluActivationWeights:o=null,leakyreluAlpha:s=0,activation:c=null}){let l=e.shape.length,u=t.shape.length,d=n?e.shape[l-2]:e.shape[l-1],f=r?t.shape[u-1]:t.shape[u-2],p=n?e.shape[l-1]:e.shape[l-2],m=r?t.shape[u-2]:t.shape[u-1],h=e.shape.slice(0,-2),g=t.shape.slice(0,-2),_=C(h),v=C(g),y=V(e.shape.slice(0,-2),t.shape.slice(0,-2)).concat([p,m]);b(d===f,()=>`Error in matMul: inner shapes (${d}) and (${f}) of Tensors with shapes ${e.shape} and ${t.shape} and transposeA=${n} and transposeB=${r} must match.`);let x=n?[_,d,p]:[_,p,d],S=r?[v,m,f]:[v,f,m],w=Z({inputs:{x:e},backend:i,attrs:{shape:x}}),T=Z({inputs:{x:t},backend:i,attrs:{shape:S}}),E=[w,T],D=Math.max(_,v),ee=n?w.shape[1]:w.shape[2],te=a!=null,O=o!=null,ne=c===`leakyrelu`,re=c==null?null:jR(c,!0),ie=te||O||ne||re!=null,ae;if((p===1||m===1)&&ee>1e3&&ie===!1){let e=w,t=T;n&&(e=ZR({inputs:{x:w},backend:i,attrs:{perm:[0,2,1]}}),E.push(e)),r&&(t=ZR({inputs:{x:T},backend:i,attrs:{perm:[0,2,1]}}),E.push(t));let a=m!==1,o=m===1,s=e;a&&(s=Z({inputs:{x:e},backend:i,attrs:{shape:[D,ee,1]}}),E.push(s));let c=m===1?2:1,l=t;o&&(l=Z({inputs:{x:t},backend:i,attrs:{shape:[D,1,ee]}}),E.push(l));let u=IR({inputs:{a:s,b:l},backend:i});ae=YR({inputs:{x:u},backend:i,attrs:{axis:c,keepDims:!0}}),E.push(u)}else{let c=Vi(e.dtype,t.dtype),l=new MR(x,S,[D,p,m],n,r,te,re,O,ne),u=[w,T];if(a!=null&&u.push(a),O&&u.push(o),ne){let e=i.makeTensorInfo([],`float32`,oi(s,`float32`));u.push(e),E.push(e)}ae=i.runWebGLProgram(l,u,c)}let oe=Z({inputs:{x:ae},backend:i,attrs:{shape:y}});E.push(ae);for(let e of E)i.disposeIntermediateTensorInfo(e);return oe}function ez(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r;return $R({a:i,b:a,transposeA:c,transposeB:l,backend:n,bias:o,preluActivationWeights:s,leakyreluAlpha:d,activation:u})}var tz={kernelName:Or,backendName:`webgl`,kernelFunc:ez},nz=`return abs(x);`;function rz(e){let{inputs:t,backend:n}=e,{x:r}=t;if(n.shouldExecuteOnCPU([r])&&r.dtype!==`complex64`){let e=dL(n.texData.get(r.dataId).values);return n.makeTensorInfo(r.shape,r.dtype,e)}let i;return i=A().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)?new eR(r.shape,nz):new BL(r.shape,nz),n.runWebGLProgram(i,[r],r.dtype)}var iz={kernelName:`Abs`,backendName:`webgl`,kernelFunc:rz},az={kernelName:Le,backendName:`webgl`,kernelFunc:kR({opSnippet:VL+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`})},oz={kernelName:Re,backendName:`webgl`,kernelFunc:kR({opSnippet:VL+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`})},sz=`return a + b;`,cz={kernelName:`Add`,backendName:`webgl`,kernelFunc:AR({opSnippet:sz,packedOpSnippet:sz,supportsComplex:!0,cpuKernelImpl:MI})},lz=class{constructor(e,t){this.outputShape=[],this.outputShape=e,this.variableNames=t.map((e,t)=>`T${t}`);let n=[];this.variableNames.forEach(e=>{n.push(`float v${e} = get${e}AtOutCoords();`)});let r=this.variableNames.map(e=>`v${e}`).join(` + `);this.userCode=`
      void main() {
        ${n.join(`
        `)}

        float result = ${r};
        setOutput(result);
      }
    `}},uz=class{constructor(e,t){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.variableNames=t.map((e,t)=>`T${t}`);let n=[];this.variableNames.forEach(e=>{n.push(`vec4 v${e} = get${e}AtOutCoords();`)});let r=this.variableNames.map(e=>`v${e}`).join(` + `);this.userCode=`
      void main() {
        ${n.join(`
        `)}

        vec4 result = ${r};
        setOutput(result);
      }
    `}};function dz(e){let{inputs:t,backend:n}=e,r=t;if(r.length===1)return gR({inputs:{x:r[0]},backend:n});if(r.length>A().getNumber(`WEBGL_MAX_TEXTURES_IN_SHADER`)){let e=Math.floor(r.length/2);return dz({inputs:[dz({inputs:r.slice(0,e),backend:n}),dz({inputs:r.slice(e),backend:n})],backend:n})}let i=r.map(e=>e.dtype).reduce((e,t)=>Vi(e,t)),a=r.map(e=>e.shape),o=A().getBool(`WEBGL_PACK`)?new uz(r[0].shape,a):new lz(r[0].shape,a);return n.runWebGLProgram(o,r,i)}var fz={kernelName:ze,backendName:`webgl`,kernelFunc:dz};function pz(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=O(a,i.shape),l=c,u=ol(l,s),d=i;u!=null&&(d=ZR({inputs:{x:i},backend:n,attrs:{perm:u}}),l=cl(l.length,s)),al(`all`,l,s);let[f,p]=rl(d.shape,l),m=C(p),h=Z({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),g=UR(h,h.dtype,`all`,n),_;if(o){let e=il(f,c);_=Z({inputs:{x:g},backend:n,attrs:{shape:e}})}else _=Z({inputs:{x:g},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(g),u!=null&&n.disposeIntermediateTensorInfo(d),_}var mz={kernelName:`All`,backendName:`webgl`,kernelFunc:pz};function hz(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=O(a,i.shape),l=c,u=ol(l,s),d=i;u!=null&&(d=ZR({inputs:{x:i},backend:n,attrs:{perm:u}}),l=cl(l.length,s)),al(`any`,l,s);let[f,p]=rl(d.shape,l),m=C(p),h=Z({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),g=UR(h,h.dtype,`any`,n),_;if(o){let e=il(f,c);_=Z({inputs:{x:g},backend:n,attrs:{shape:e}})}else _=Z({inputs:{x:g},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(g),u!=null&&n.disposeIntermediateTensorInfo(d),_}var gz={kernelName:`Any`,backendName:`webgl`,kernelFunc:hz},_z=class{constructor(e,t,n){this.variableNames=[`A`];let{windowSize:r,batchSize:i,outSize:a}=e;n||this.variableNames.push(`bestIndicesA`),this.outputShape=[i,a];let o=t===`max`?`>`:`<`,s=n?`inOffset + i;`:`round(getBestIndicesA(batch, inOffset + i));`;this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${r};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${r}; i++) {
          int inIdx = ${s};
          float candidate = getA(batch, inIdx);
          if (candidate ${o} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}},vz=class{constructor(e,t,n,r){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,b(e.length>2,()=>`Packed arg${n.charAt(0).toUpperCase()+n.slice(1)} supports only inputs with rank above 2.`);let i=e[e.length-1],a=Math.ceil(i/t);this.outputShape=e.slice(0,-1),a>1&&this.outputShape.push(a),r||this.variableNames.push(`bestIndicesA`);let o=this.outputShape,s=o.length,c=GF(s),l=OL(`coords`,s),u,d;if(a===1){d=s+1;let e=GF(d);u=`
        ${e} sourceLocR = ${e}(${l.join()}, 0);
        ++${l[s-1]};
        ${e} sourceLocG = ${e}(${l.join()}, 0);
        ++${l[s-2]};
        ${e} sourceLocA = ${e}(${l.join()}, 0);
        --${l[s-1]};
        ${e} sourceLocB = ${e}(${l.join()}, 0);
        --${l[s-2]};`}else d=s,u=`
        ${c} sourceLocR = coords;
        ++${l[s-1]};
        ${c} sourceLocG = coords;
        ++${l[s-2]};
        ${c} sourceLocA = coords;
        --${l[s-1]};
        ${c} sourceLocB = coords;
        --${l[s-2]};`;let f=[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,d),p=`.`+f[d-1],m=f.map(e=>`int `+e),h=OL(`sourceLocR`,d-1).concat(`inIdx.r`),g=OL(`sourceLocG`,d-1).concat(`inIdx.g`),_=OL(`sourceLocB`,d-1).concat(`inIdx.b`),v=OL(`sourceLocA`,d-1).concat(`inIdx.a`),y=n===`max`?`greaterThan`:`lessThan`,x=r?``:`
          inIdx = round(vec4(getBestIndicesAChannel(${h.join()}),
                             getBestIndicesAChannel(${g.join()}),
                             getBestIndicesAChannel(${_.join()}),
                             getBestIndicesAChannel(${v.join()})));`,S=`vec4(
            getAChannel(${h.join()}),
            hasNextCol ? getAChannel(${g.join()}) : 0.,
            hasNextRow ? getAChannel(${_.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${v.join()}) : 0.)`,C=r?``:`
      float getBestIndicesAChannel(${m.join()}) {
        return getChannel(getBestIndicesA(${f.join()}),
                                          vec2(${f.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${m.join()}) {
        return getChannel(getA(${f.join()}),
                               vec2(${f.slice(-2).join()}));
      }
      ${C}
      void main() {
        ${c} coords = getOutputCoords();
        bool hasNextCol = ${l[s-1]} < ${o[s-1]-1};
        bool hasNextRow = ${l[s-2]} < ${o[s-2]-1};
        ${u}
        ivec4 srcIdx = ivec4(sourceLocR${p}, sourceLocG${p},
          sourceLocB${p}, sourceLocA${p}) * ${t};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${S};

        for (int i = 0; i < ${t}; i++) {
          inIdx = srcIdx;
          ${x}
          vec4 candidate = ${S};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${y}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}};function yz(e,t,n,r=null){let i=t.shape[0],a=t.shape[1];r!=null&&(i=r.shape[0],a=r.shape[1]);let o=Th(a),s=new _z({windowSize:o,inSize:a,batchSize:i,outSize:Math.ceil(a/o)},n,r==null),c=[t];r!=null&&c.push(r);let l=e.runWebGLProgram(s,c,`int32`);if(l.shape[1]===1)return l;let u=yz(e,t,n,l);return e.disposeIntermediateTensorInfo(l),u}function bz(e,t,n,r=null){let i=r==null?t.shape:r.shape,a=i[i.length-1],o=new vz(i,Th(a),n,r==null),s=r==null?[t]:[t,r],c=e.runWebGLProgram(o,s,`int32`);if(c.shape.length===t.shape.length){let r=bz(e,t,n,c);return e.disposeIntermediateTensorInfo(c),r}return c}function xz(e,t,n,r){let i=[n];if(al(`arg`+r.charAt(0).toUpperCase()+r.slice(1),i,t.shape.length),!A().getBool(`WEBGL_PACK_REDUCE`)||t.shape.length<=2){let n=[],a=e.texData.get(t.dataId),o=a!==null&&a.isPacked,s=t;o&&(s=e.unpackTensor(t),n.push(s));let[c,l]=rl(s.shape,i),u=C(l),d=Z({inputs:{x:s},backend:e,attrs:{shape:[-1,u]}});n.push(d);let f=yz(e,d,r);n.push(f);let p=Z({inputs:{x:f},backend:e,attrs:{shape:c}});return n.forEach(t=>e.disposeIntermediateTensorInfo(t)),p}return bz(e,t,r)}function Sz(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r,o=O(a,i.shape),s=ol(o,i.shape.length),c=i,l=[];s!=null&&(c=ZR({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=cl(o.length,c.shape.length)),al(`argMax`,[o[0]],c.shape.length);let u=xz(n,c,o[0],`max`);return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),u}var Cz={kernelName:Be,backendName:`webgl`,kernelFunc:Sz};function wz(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r,o=O(a,i.shape),s=ol(o,i.shape.length),c=i,l=[];s!=null&&(c=ZR({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=cl(o.length,c.shape.length)),al(`argMin`,[o[0]],c.shape.length);let u=xz(n,c,o[0],`min`);return l.forEach(e=>n.disposeIntermediateTensorInfo(e)),u}var Tz={kernelName:Ve,backendName:`webgl`,kernelFunc:wz},Ez={kernelName:He,backendName:`webgl`,kernelFunc:kR({opSnippet:VL+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`})},Dz={kernelName:Ue,backendName:`webgl`,kernelFunc:kR({opSnippet:VL+`return log(x + sqrt(x * x + 1.0));`})},Oz={kernelName:We,backendName:`webgl`,kernelFunc:kR({opSnippet:VL+`
  return atan(x);
`})},kz={kernelName:Ke,backendName:`webgl`,kernelFunc:AR({opSnippet:fR+`
  return atan(a, b);
`,packedOpSnippet:`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+mR+`
  return result;
`})},Az={kernelName:Ge,backendName:`webgl`,kernelFunc:kR({opSnippet:VL+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`})},jz=class{constructor(e,t,n,r=!1,i=!1){if(this.variableNames=[`x`],t===`avg`&&n)throw Error(`Cannot compute positions for average pool.`);let a=e.filterWidth,o=e.strideHeight,s=e.strideWidth,c=e.dilationHeight,l=e.dilationWidth,u=e.effectiveFilterHeight,d=e.effectiveFilterWidth,f=e.padInfo.top,p=e.padInfo.left;this.outputShape=e.outShape;let m=t===`avg`,h=`((batch  * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + d`,g=`(xR * ${e.inWidth} + xC) * ${e.inChannels} + d`,_=`0.0`;if(m||(_=`-1.0 / 1e-20`),n){this.userCode=`
        const ivec2 strides = ivec2(${o}, ${s});
        const ivec2 pads = ivec2(${f}, ${p});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${u};
              wR += ${c}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${d};
                wC += ${l}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value >= currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${r?i?h:g:`wR * ${d} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}let v=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`avg`&&(v=`avgValue / max(count, 1.0)`);let y=Math.floor(a/4)*4,b=a%4,x=`
      if (${m}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = max(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${o}, ${s});
      const ivec2 pads = ivec2(${f}, ${p});
      const float initializationValue = ${_};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${_});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${u};
            wR += ${c}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${y}; wC += 4) {
            int xC = xCCorner + wC * ${l};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              getValue(batch, xR, xC + 2 * ${l}, d),
              getValue(batch, xR, xC + 3 * ${l}, d)
            );

            ${x}
          }

          int xC = xCCorner + ${y};
          if (${b===1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${x}
          } else if (${b===2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              initializationValue,
              initializationValue
            );

            ${x}
          } else if (${b===3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${l}, d),
              getValue(batch, xR, xC + 2 * ${l}, d),
              initializationValue
            );

            ${x}
          }
        }
        setOutput(${v});
      }
    `}},Mz=class{constructor(e,t,n,r=!1,i=!1){if(this.variableNames=[`x`],t===`avg`&&n)throw Error(`Cannot compute positions for average pool.`);let a=e.filterWidth,o=e.strideDepth,s=e.strideHeight,c=e.strideWidth,l=e.dilationDepth,u=e.dilationHeight,d=e.dilationWidth,f=e.effectiveFilterDepth,p=e.effectiveFilterHeight,m=e.effectiveFilterWidth,h=e.padInfo.front,g=e.padInfo.top,_=e.padInfo.left;this.outputShape=e.outShape;let v=t===`avg`,y=`0.0`;if(v||(y=`-1.0 / 1e-20`),n){this.userCode=`
        const ivec3 strides =
            ivec3(${o}, ${s}, ${c});
        const ivec3 pads = ivec3(${h}, ${g}, ${_});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${f};
              wD += ${l}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${e.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${p};
                wR += ${u}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${m};
                  wC += ${d}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value >= currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${r?i?`(((batch * ${e.inDepth} + xD) * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`((xD * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`wD * ${p} * ${m} +
                      wR * ${m} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}let b=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t===`avg`&&(b=`avgValue / max(count, 1.0)`);let x=Math.floor(a/4)*4,S=a%4,C=`
      if (${v}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = max(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${o}, ${s}, ${c});
      const ivec3 pads = ivec3(${h}, ${g}, ${_});
      const float initializationValue = ${y};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${y});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${f};
            wD += ${l}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${p};
            wR += ${u}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${x}; wC += 4) {
              int xC = xCCorner + wC * ${d};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                getValue(batch, xD, xR, xC + 3 * ${d}, ch)
              );

              ${C}
            }

            int xC = xCCorner + ${x};
            if (${S===1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${C}
            } else if (${S===2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                initializationValue,
                initializationValue
              );

              ${C}
            } else if (${S===3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                initializationValue
              );

              ${C}
            }
          }
        }
        setOutput(${b});
      }
    `}};function Nz(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;JP(i,`avgPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;b(ms(o,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=es(i.shape,a,o,1,s,c);if(l.filterWidth===1&&l.filterHeight===1&&w(l.inShape,l.outShape))return gR({inputs:{x:i},backend:n});let u=new jz(l,`avg`,!1);return n.runWebGLProgram(u,[i],`float32`)}var Pz={kernelName:qe,backendName:`webgl`,kernelFunc:Nz};function Fz(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c,dataFormat:l}=r,u=new Mz(ts(i.shape,a,o,[1,1,1],s,c,l),`avg`,!1);return n.runWebGLProgram(u,[i],`float32`)}var Iz={kernelName:Ye,backendName:`webgl`,kernelFunc:Fz},Lz=class{constructor(e){this.variableNames=[`dy`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=e.dilationHeight,o=e.dilationWidth,s=e.effectiveFilterHeight,c=e.effectiveFilterWidth,l=s-1-e.padInfo.top,u=c-1-e.padInfo.left,d=1/(t*n);this.userCode=`
      const ivec2 pads = ivec2(${l}, ${u});
      const float avgMultiplier = float(${d});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${s};
            wR += ${a}) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${c};
            wC+= ${o}) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}},Rz=class{constructor(e){this.variableNames=[`dy`],this.outputShape=e.inShape;let t=e.filterDepth,n=e.filterHeight,r=e.filterWidth,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=e.dilationDepth,c=e.dilationHeight,l=e.dilationWidth,u=e.effectiveFilterDepth,d=e.effectiveFilterHeight,f=e.effectiveFilterWidth,p=u-1-e.padInfo.front,m=d-1-e.padInfo.top,h=f-1-e.padInfo.left,g=1/(t*n*r);this.userCode=`
      const ivec3 pads = ivec3(${p}, ${m}, ${h});
      const float avgMultiplier = float(${g});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${u};
            wD += ${s}) {
          float dyD = float(dyDCorner + wD) / ${i}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${d};
              wR += ${c}) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${f};
                wC += ${l}) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function zz(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a,{filterSize:s,strides:c,pad:l,dimRoundingMode:u}=r,d=new Rz(ts(o.shape,s,c,[1,1,1],l,u));return n.runWebGLProgram(d,[i],o.dtype)}var Bz={kernelName:Xe,backendName:`webgl`,kernelFunc:zz};function Vz(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a;JP([i,a],`avgPoolGrad`);let{filterSize:s,strides:c,pad:l}=r,u=new Lz(es(o.shape,s,c,1,l));return n.runWebGLProgram(u,[i],o.dtype)}var Hz={kernelName:Je,backendName:`webgl`,kernelFunc:Vz};function Uz(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;return $R({a:i,b:a,transposeA:o,transposeB:s,backend:n})}var Wz={kernelName:Ze,backendName:`webgl`,kernelFunc:Uz},Gz=class{constructor(e,t,n,r,i,a){this.outputShape=[],this.variableNames=[`x`,`mean`,`variance`],V(e,t),V(e,n);let o=`0.0`;r!=null&&(V(e,r),this.variableNames.push(`offset`),o=`getOffsetAtOutCoords()`);let s=`1.0`;i!=null&&(V(e,i),this.variableNames.push(`scale`),s=`getScaleAtOutCoords()`),this.outputShape=e,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${o};
        float scale = ${s};
        float inv = scale * inversesqrt(variance + float(${a}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}},Kz=class{constructor(e,t,n,r,i,a){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=[`x`,`mean`,`variance`],V(e,t),V(e,n);let o=`vec4(0.0)`;r!=null&&(V(e,r),this.variableNames.push(`offset`),o=`getOffsetAtOutCoords()`);let s=`vec4(1.0)`;i!=null&&(V(e,i),this.variableNames.push(`scale`),s=`getScaleAtOutCoords()`),this.outputShape=e,this.userCode=`
      void main() {
        vec4 offset = ${o};
        vec4 scale = ${s};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${a}));

        setOutput((x - mean) * inv + offset);
      }
    `}},qz={kernelName:zt,backendName:`webgl`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r,mean:i,variance:a,offset:o,scale:s}=e;b(i.shape.length===a.shape.length,()=>`Batch normalization gradient requires mean and variance to have equal ranks.`),b(o==null||i.shape.length===o.shape.length,()=>`Batch normalization gradient requires mean and offset to have equal ranks.`),b(s==null||i.shape.length===s.shape.length,()=>`Batch normalization gradient requires mean and scale to have equal ranks.`);let{varianceEpsilon:c}=n;c??=.001;let l=[r,i,a],u=null;o!=null&&(u=o.shape,l.push(o));let d=null;s!=null&&(d=s.shape,l.push(s));let f=A().getBool(`WEBGL_PACK_NORMALIZATION`)?new Kz(r.shape,i.shape,a.shape,u,d,c):new Gz(r.shape,i.shape,a.shape,u,d,c);return t.runWebGLProgram(f,l,l[0].dtype)}},Jz=class{constructor(e){this.variableNames=[`source`],this.outputShape=e,this.rank=e.length;let t=GF(this.rank);this.customUniforms=[{name:`start`,arrayIndex:this.rank,type:`int`}];let n=Xz(this.rank),r;r=`
        ${t} sourceLoc;
        ${t} coords = getOutputCoords();
        ${e.map((e,t)=>`sourceLoc.${Yz[t]} = start[${t}] + coords.${Yz[t]};`).join(`
`)}
      `,this.userCode=`
      void main() {
        ${r}
        setOutput(getSource(${n}));
      }
    `}},Yz=[`x`,`y`,`z`,`w`,`u`,`v`];function Xz(e){if(e===1)return`sourceLoc`;if(e<=6)return Yz.slice(0,e).map(e=>`sourceLoc.`+e).join(`,`);throw Error(`Slicing for rank ${e} is not yet supported`)}var Zz=class{constructor(e){this.variableNames=[`source`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.customUniforms=[{name:`start`,arrayIndex:this.rank,type:`int`}];let t=GF(this.rank),n=OL(`coords`,this.rank),r=OL(`sourceLoc`,this.rank),i=this.rank===1?`sourceLoc`:`vec2(${r.slice(-2).join()})`,a=`getChannel(getSource(${r.join()}), ${i})`,o=`
      result.x = ${a};
      if (++${n[this.rank-1]} < ${e[this.rank-1]}) {
        ++${r[this.rank-1]};
        result.y = ${a};
        --${r[this.rank-1]};
      }
    `,s=this.rank===1?``:`
      --${n[this.rank-1]};
      if (++${n[this.rank-2]} < ${e[this.rank-2]}) {
        ++${r[this.rank-2]};
        result.z = ${a};
        if (++${n[this.rank-1]} < ${e[this.rank-1]}) {
          ++${r[this.rank-1]};
          result.w = ${a};
        }
      }
    `,c=this.rank<=4?`sourceLoc = coords +
            ${t}(${e.map((e,t)=>`start[${t}]`).join()});`:e.map((e,t)=>`${r[t]} = ${n[t]} + start[${t}];`).join(`
`);this.userCode=`
      void main() {
        ${t} coords = getOutputCoords();
        ${t} sourceLoc;
        ${c}
        vec4 result = vec4(0.);
        ${o}
        ${s}
        setOutput(result);
      }
    `}};function Qz(e,t,n,r){let i=r.texData.get(e.dataId),a=r.makeTensorInfo(n,e.dtype),o=r.texData.get(a.dataId);Object.assign(o,i),o.refCount=1,o.shape=n,o.dtype=e.dtype;let s=uh(t,k(e.shape));i.slice&&(s+=i.slice.flatOffset),o.slice={flatOffset:s,origDataId:i.slice&&i.slice.origDataId||e.dataId};let c=r.dataRefCount.get(o.slice.origDataId)||1;return r.dataRefCount.set(o.slice.origDataId,c+1),a}function $z(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,size:o}=r,[s,c]=dh(i,a,o);if(Zm(i,s,c),C(c)===0)return n.makeTensorInfo(c,i.dtype,[]);if(n.shouldExecuteOnCPU([i])||i.dtype===`string`){let e=fL(n.texData.get(i.dataId).values,s,c,i.shape,i.dtype);return n.makeTensorInfo(c,i.dtype,e)}let{isPacked:l}=n.texData.get(i.dataId),u=lh(i.shape,s,c);if(l||!u){let e=A().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new Zz(c):new Jz(c),t=[s];return n.runWebGLProgram(e,[i],i.dtype,t)}return n.uploadToGPU(i.dataId),Qz(i,s,c,n)}var eB={kernelName:Yn,backendName:`webgl`,kernelFunc:$z},tB={kernelName:Qe,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r;b(i.shape.length<=4,()=>`batchToSpaceND for rank > 4 with a WebGL backend not implemented yet`);let s=a.reduce((e,t)=>e*t),c=Dh(i.shape,a,s),l=Oh(c.length,a.length),u=kh(i.shape,a,s),d=Ah(o,a.length),f=jh(u,o,a.length),p=[],m=Z({inputs:{x:i},backend:n,attrs:{shape:c}}),h=ZR({inputs:{x:m},backend:n,attrs:{perm:l}}),g=Z({inputs:{x:h},backend:n,attrs:{shape:u}}),_=$z({inputs:{x:g},backend:n,attrs:{begin:d,size:f}});return p.push(m),p.push(h),p.push(g),p.forEach(e=>n.disposeIntermediateTensorInfo(e)),_}};function nB(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o}=r,s=NI(n.readSync(i.dataId),n.readSync(a.dataId),a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,s)}var rB={kernelName:$e,backendName:`webgl`,kernelFunc:nB},iB=`
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`,aB=`
  return float(int(a.r) & int(b.r));
`;function oB(e){let{inputs:t,backend:n}=e,{a:r,b:i}=t,a=A().getBool(`WEBGL_PACK_BINARY_OPERATIONS`),o=A().getNumber(`WEBGL_VERSION`);if(n.shouldExecuteOnCPU([r,i])||o===1){let e=n.texData.get(r.dataId).values,t=n.texData.get(i.dataId).values,[a,o]=FI(r.shape,i.shape,e,t,r.dtype),s=n.makeTensorInfo(o,r.dtype),c=n.texData.get(s.dataId);return c.values=a,s}let s;return s=a?new hR(iB,r.shape,i.shape,!1):new pR(aB,r.shape,i.shape),n.runWebGLProgram(s,[r,i],r.dtype)}var sB={kernelName:et,backendName:`webgl`,kernelFunc:oB};function cB(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t,a=n.readSync(r.dataId),o=n.readSync(i.dataId),s=V(Array.from(a),Array.from(o));return n.makeTensorInfo([s.length],`int32`,Int32Array.from(s))}var lB={kernelName:nt,backendName:`webgl`,kernelFunc:cB},uB=AR({opSnippet:`return float(a != b);`,cpuKernelImpl:nL,dtype:`bool`}),dB={kernelName:vn,backendName:`webgl`,kernelFunc:uB};function fB(e){let{inputs:t,backend:n}=e,{input:r}=t;return gR({inputs:{x:n.texData.get(r.dataId).complexTensorInfos.real},backend:n})}var pB={kernelName:Mn,backendName:`webgl`,kernelFunc:fB},mB=`return float(int(x));`;function hB(e,t){let n=new BL(e.shape,mB),r=t.runWebGLProgram(n,[e],`int32`);return{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}function gB(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dtype:a}=r;if(a===`complex64`){if(i.dtype===`complex64`)return gR({inputs:{x:i},backend:n});let e=Ru(i.shape),t=gB({inputs:{x:i},backend:n,attrs:{dtype:`float32`}}),r=vR({inputs:{real:t,imag:e},backend:n});return e.dispose(),n.disposeIntermediateTensorInfo(t),r}if(i.dtype===`complex64`){let e=fB({inputs:{input:i},backend:n}),t=gB({inputs:{x:e},backend:n,attrs:{dtype:a}});return n.disposeIntermediateTensorInfo(e),t}if(!se(i.dtype,a)){let e=gR({inputs:{x:i},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:a}}if(n.shouldExecuteOnCPU([i])){let e=n.texData.get(i.dataId).values,[t,r,o]=II(e,i.shape,i.dtype,a);return n.makeTensorInfo(t,r,o)}if(a===`int32`)return hB(i,n);if(a===`bool`){let e=n.makeTensorInfo([],`bool`,re(`bool`,1)),t=uB({inputs:{a:i,b:e},backend:n});return n.disposeIntermediateTensorInfo(e),t}throw Error(`Error in Cast: failed to cast ${i.dtype} to ${a}`)}var _B={kernelName:rt,backendName:`webgl`,kernelFunc:gB},vB=`return ceil(x);`,yB={kernelName:it,backendName:`webgl`,kernelFunc:kR({opSnippet:vB,packedOpSnippet:vB,cpuKernelImpl:LI})},bB=class{constructor(e){this.variableNames=[`A`],this.customUniforms=[{name:`minVal`,type:`float`},{name:`maxVal`,type:`float`}],this.outputShape=e,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}},xB=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`minVal`,type:`float`},{name:`maxVal`,type:`float`}],this.outputShape=e,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}};function SB(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{clipValueMin:a,clipValueMax:o}=r,s;s=A().getBool(`WEBGL_PACK_CLIP`)?new xB(i.shape):new bB(i.shape);let c=[[a],[o]];return n.runWebGLProgram(s,[i],i.dtype,c)}var CB={kernelName:at,backendName:`webgl`,kernelFunc:SB},wB=class{constructor(e){this.variableNames=[`real`,`imag`],this.outputShape=e,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}};function TB(e,t){return{dataId:t.dataId,dtype:t.dtype,shape:e.shape}}function EB(e){let{inputs:t,backend:n}=e,{x:r}=t,i=n.texData.get(r.dataId),a=new wB(r.shape),o=[TB(r,i.complexTensorInfos.real),TB(r,i.complexTensorInfos.imag)];return n.runWebGLProgram(a,o,o[0].dtype)}var DB={kernelName:st,backendName:`webgl`,kernelFunc:EB},OB=class{constructor(e){this.outputShape=[],this.outputShape=yh(e,1),this.variableNames=e.map((e,t)=>`T${t}`);let t=Array(e.length-1);t[0]=e[0][1];for(let n=1;n<t.length;n++)t[n]=t[n-1]+e[n][1];let n=[`if (yC < ${t[0]}) setOutput(getT0(yR, yC));`];for(let e=1;e<t.length;e++){let r=t[e-1];n.push(`else if (yC < ${t[e]}) setOutput(getT${e}(yR, yC-${r}));`)}let r=t.length,i=t[t.length-1];n.push(`else setOutput(getT${r}(yR, yC-${i}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${n.join(`
        `)}
      }
    `}},kB=class{constructor(e,t){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=yh(e,t);let n=this.outputShape,r=n.length,i=GF(r),a=OL(`coords`,r),o=[`x`,`y`,`z`,`w`,`u`,`v`].slice(0,r);this.variableNames=e.map((e,t)=>`T${t}`);let s=Array(e.length-1);s[0]=e[0][t];for(let n=1;n<s.length;n++)s[n]=s[n-1]+e[n][t];let c=o[t],l=o.slice(-2),u=o.join(),d=`if (${c} < ${s[0]}) {
        return getChannel(
            getT0(${u}), vec2(${l.join()}));
        }`;for(let e=1;e<s.length;e++){let t=s[e-1];d+=`
        if (${c} < ${s[e]}  && ${c} >= ${s[e-1]}) {
          return getChannel(
            getT${e}(${AB(o,c,t)}),
            vec2(${AB(l,c,t)}));
        }`}let f=s.length,p=s[s.length-1];d+=`
        return getChannel(
          getT${f}(${AB(o,c,p)}),
          vec2(${AB(l,c,p)}));`,this.userCode=`
      float getValue(${o.map(e=>`int `+e)}) {
        ${d}
      }

      void main() {
        ${i} coords = getOutputCoords();
        vec4 result = vec4(getValue(${a}), 0., 0., 0.);

        ${a[r-1]} = ${a[r-1]} + 1;
        if (${a[r-1]} < ${n[r-1]}) {
          result.g = getValue(${a});
        }

        ${a[r-2]} = ${a[r-2]} + 1;
        if (${a[r-2]} < ${n[r-2]}) {
          result.a = getValue(${a});
        }

        ${a[r-1]} = ${a[r-1]} - 1;
        if (${a[r-2]} < ${n[r-2]} &&
            ${a[r-1]} < ${n[r-1]}) {
          result.b = getValue(${a});
        }
        setOutput(result);
      }
    `}};function AB(e,t,n){let r=e.indexOf(t);return e.map((e,t)=>t===r?`${e} - ${n}`:e).join()}function jB(e){let{inputs:t,backend:n}=e,{input:r}=t;return gR({inputs:{x:n.texData.get(r.dataId).complexTensorInfos.imag},backend:n})}var MB={kernelName:Kt,backendName:`webgl`,kernelFunc:jB};function NB(e,t,n){let r=e[0].dtype;if(r===`complex64`){let r=e.map(e=>fB({inputs:{input:e},backend:n})),i=e.map(e=>jB({inputs:{input:e},backend:n})),a=NB(r,t,n),o=NB(i,t,n),s=vR({inputs:{real:a,imag:o},backend:n});return r.forEach(e=>n.disposeIntermediateTensorInfo(e)),i.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.disposeIntermediateTensorInfo(a),n.disposeIntermediateTensorInfo(o),s}let i=n.shouldExecuteOnCPU(e);if(r===`string`&&(i=!0),i){let i=e.map(e=>{let r=[-1,C(e.shape.slice(t))];return Z({inputs:{x:e},backend:n,attrs:{shape:r}})}),a=RI(i.map(e=>({vals:n.readSync(e.dataId),shape:e.shape})),yh(i.map(e=>e.shape),1),r,i[0].shape[0]===1),o=yh(e.map(e=>e.shape),t),s=n.makeTensorInfo(o,r,a);return i.forEach(e=>n.disposeIntermediateTensorInfo(e)),s}let a=e.filter(e=>C(e.shape)>0),o=A().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)&&a[0].shape.length>1;if(a.length===1){let t=o?new BL(e[0].shape,qL):new eR(e[0].shape,qL);return n.runWebGLProgram(t,e,r)}let s=A().getNumber(`WEBGL_MAX_TEXTURES_IN_SHADER`);if(a.length>s){let e=[];for(let r=0;r<a.length;r+=s){let i=a.slice(r,r+s);e.push(NB(i,t,n))}let r=NB(e,t,n);for(let t of e)n.disposeIntermediateTensorInfo(t);return r}if(o){let e=new kB(a.map(e=>e.shape),t);return n.runWebGLProgram(e,a,r)}let{tensors2D:c,outShape:l}=PB(a,t,n),u=new OB(c.map(e=>e.shape)),d=n.runWebGLProgram(u,c,r);c.forEach(e=>n.disposeIntermediateTensorInfo(e));let f=Z({inputs:{x:d},attrs:{shape:l},backend:n});return n.disposeIntermediateTensorInfo(d),f}function PB(e,t,n){let r=yh(e.map(e=>e.shape),t);return{tensors2D:e.map(e=>Z({inputs:{x:e},attrs:{shape:[-1,C(e.shape.slice(t))]},backend:n})),outShape:r}}function FB(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r,a=O(i,t[0].shape)[0];vh(t.map(e=>e.shape),a);let o=yh(t.map(e=>e.shape),a);if(C(o)===0)return n.makeTensorInfo(o,t[0].dtype,[]);let s=t.filter(e=>C(e.shape)>0);return s.length===1?gR({inputs:{x:s[0]},backend:n}):NB(s,a,n)}var IB={kernelName:ct,backendName:`webgl`,kernelFunc:FB},LB=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let a=e.padInfo.top,o=e.padInfo.left,s=e.strideHeight,c=e.strideWidth,l=e.dilationHeight,u=e.dilationWidth,d=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,m=e.inChannels%4,h=e.dataFormat===`channelsLast`,g=h?1:2,_=h?2:3,v=h?3:1,y=``,b=``;n&&(y=r?`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`
          float activation(float x) {
            ${n}
          }
        `,b=`result = activation(result);`);let x=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${y}

      const ivec2 strides = ivec2(${s}, ${c});
      const ivec2 pads = ivec2(${a}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${v}];

        ivec2 xRCCorner =
            ivec2(coords[${g}], coords[${_}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${d}; wR++) {
          int xR = xRCorner + wR * ${l};

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${f}; wC++) {
            int xC = xCCorner + wC * ${u};

            if (xC < 0 || xC >= ${e.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${p}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${h}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${m===1}) {

              if (${h}) {
                dotProd +=
                    getX(batch, xR, xC, ${p}) *
                    getW(wR, wC, ${p}, d2);
              } else {
                dotProd +=
                    getX(batch, ${p}, xR, xC) *
                    getW(wR, wC, ${p}, d2);
              }

            } else if (${m===2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2)
              );

              if (${h}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${m===3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2),
                getW(wR, wC, ${p} + 2, d2)
              );

              if (${h}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1),
                  getX(batch, xR, xC, ${p} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC),
                  getX(batch, ${p} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${x}
        ${b}
        setOutput(result);
      }
    `}},RB=class{constructor(e){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let t=e.padInfo.front,n=e.padInfo.top,r=e.padInfo.left,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=e.dilationDepth,c=e.dilationHeight,l=e.dilationWidth,u=e.filterDepth,d=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,m=e.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${i}, ${a}, ${o});
      const ivec3 pads = ivec3(${t}, ${n}, ${r});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${u}; wF++) {
          int xF = xFCorner + wF * ${s};

          if (xF < 0 || xF >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${d}; wR++) {
            int xR = xRCorner + wR * ${c};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${f}; wC++) {
              int xC = xCCorner + wC * ${l};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${p}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${m===1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${p}) *
                  getW(wF, wR, wC, ${p}, d2);
              } else if (${m===2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${m===3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1),
                  getX(batch, xF, xR, xC, ${p} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2),
                  getW(wF, wR, wC, ${p} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}},zB=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=eI(this.outputShape.length);let a=e.padInfo.left,o=e.strideWidth,s=e.dilationWidth,c=e.filterHeight,l=e.filterWidth,u=l,d=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let e=0;e<l;e++)d+=`
           vec4 xTexelC${e*2};
           int xTexelC${e*2}Ready;
           vec4 xTexelC${e*2+1};
           int xTexelC${e*2+1}Ready;
           vec4 xC${e};`;d+=`
     for (int r = 0; r < ${c}; r++) {
      for (int d1 = 0; d1 < ${e.inChannels}; d1 += 2) {
       `;for(let e=0;e<l;e++)d+=`
           xTexelC${e*2} = vec4(0.0);
           xTexelC${e*2}Ready = 0;
           xTexelC${e*2+1} = vec4(0.0);
           xTexelC${e*2+1}Ready = 0;
           xC${e} = vec4(0.0);`;d+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let t=0;t<(u+1)/2;t++){let n=t*2;if(d+=`
           xC = xCCorner + ${n*s};
           `,o===1){if(n<l&&(a%2==1?(d+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }
               `,d+=s===1&&n>0?`
                 xC${n} = vec4(xTexelC${n-2}.zw, xTexelC${n}.xy);
                 `:`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${n} = vec4(previous.zw, xTexelC${n}.xy);
                   } else {
                     xC${n} = vec4(0.0, 0.0, xTexelC${n}.xy);
                   }
                   `):d+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 xC${n} = xTexelC${n};
                 `,n+1<l)){let e=a%2==0?_(s):s;s%2==0&&a%2==1||s%2!=0&&a%2!=1?(d+=`
                   xCOffset = xC + imod(pads[1], 2) + ${e};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                     xTexelC${n+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${n+1}.zw = vec2(0.0);
                     }
                     xTexelC${n+1}Ready = 1;
                   }
                   `,d+=s>1?`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${n+1} = vec4(previous.zw, xTexelC${n+1}.xy);
                     } else {
                      xC${n+1} = vec4(0.0, 0.0, xTexelC${n+1}.xy);
                     }
                     `:`
                     xC${n+1} = vec4(xTexelC${n}.zw, xTexelC${n+1}.xy);
                     `):d+=e===1?`
                     xC${n+1} = xTexelC${n};
                     `:`
                     xCOffset = xC + ${e};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                       xTexelC${n+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${n+1}.zw = vec2(0.0);
                       }
                       xTexelC${n+1}Ready = 1;
                     }

                     xC${n+1} = xTexelC${n+1};
                     `}}else n<l&&(a%2==1?(d+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${n+1}Ready == 0) {
                   xTexelC${n+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${n+1}.zw = vec2(0.0);
                   }
                   xTexelC${n+1}Ready = 1;
                 }

                 xC${n} = vec4(xTexelC${n}.zw, xTexelC${n+1}.zw);
               `,n+1<l&&(d+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${n+1} = vec4(xTexelC${n+1}.xy, final.xy);
                 `)):(d+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${n}Ready == 0) {
                   xTexelC${n} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${n}.zw = vec2(0.0);
                   }
                   xTexelC${n}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${n+1}Ready == 0) {
                   xTexelC${n+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${n+1}.zw = vec2(0.);
                   }
                   xTexelC${n+1}Ready = 1;
                 }

                 xC${n} = vec4(
                   xTexelC${n}.xy, xTexelC${n+1}.xy);
               `,n+1<l&&(d+=`
                   xC${n+1} = vec4(xTexelC${n}.zw, xTexelC${n+1}.zw);
                 `)));n<l&&(d+=`
             wTexel = getW(r, ${n}, d1, d2);
             dotProd += xC${n}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${e.inChannels}) {
               dotProd += xC${n}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,n+1<l&&(d+=`
               wTexel = getW(r, ${n+1}, d1, d2);
               dotProd += xC${n+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${e.inChannels}) {
                 dotProd += xC${n+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}d+=`
     }
   `,d+=`
     }
   `,d+=`
     }
   `;let f=``,p=``;n&&(f=r?`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${n}
         }`:i?`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${n}
         }`:`vec4 activation(vec4 x) {
           ${n}
         }`,p=`result = activation(result);`);let m=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
       ${f}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${d}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${m}
         ${p}
         setOutput(result);
       }
     `}},BB=class{constructor(e,t){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`inputShape`,type:`ivec4`},{name:`pad`,type:`ivec2`},{name:`stride`,type:`ivec2`},{name:`dilation`,type:`ivec2`},{name:`inChannels`,type:`int`},{name:`itemsPerBlockRow`,type:`int`},{name:`outWidth`,type:`int`}],this.outputShape=e,this.enableShapeUniforms=eI(this.outputShape.length);let{dataFormat:n}=t,r=YP(),i=n===`channelsLast`,a=i?1:2,o=i?2:3,s=this.enableShapeUniforms?`if(blockIndex < outShape[2] && pos < outShape[1]) {`:`if(blockIndex < ${e[2]} && pos < ${e[1]}) {`,c=``;for(let e=0;e<=1;e++)for(let t=0;t<=1;t++)c+=`
          blockIndex = rc.z + ${t};
          pos = rc.y + ${e};

          ${s}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${a}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${o}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${i}) {
                  innerDims = vec2(d1, ch);
                  result[${e*2+t}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${e*2+t}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${c}

        ${r.output} = result;
      }
    `}};function VB(e,t){let n=e.length;return n>=3?t?[...e.slice(0,-3),e[n-3]*e[n-2],e[n-1]]:[...e.slice(0,-3),e[n-3],e[n-2]*e[n-1]]:!t&&n===1&&e[0]>1?[e[0],1]:null}function HB({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let c=e.shape,l=r.texData.get(e.dataId),u=n.inChannels,d=c[0]*c[1]*c[2],f=n.outChannels,p=n.dataFormat===`channelsLast`,m,h=[];if(a!=null){let e=VB(a.shape,p);e!=null&&(a=Z({inputs:{x:a},backend:r,attrs:{shape:e}}),h.push(a))}if(i!=null){let e=VB(i.shape,p);e!=null&&(i=Z({inputs:{x:i},backend:r,attrs:{shape:e}}),h.push(i))}if(!((d===1||f===1)&&u>1e3)&&l.isPacked&&p&&l.texture!=null&&c[2]%2!=0&&w(l.shape.slice(-3),c.slice(-3))){let u=c[0]*c[1]*(c[2]+1),d={dataId:e.dataId,shape:[1,u,n.inChannels],dtype:e.dtype},f=l.shape;l.shape=l.shape.slice(),l.shape[l.shape.length-2]++,b(FP(l.shape,d.shape),()=>`packed reshape ${l.shape} to ${d.shape} isn't free`);let p=Z({inputs:{x:t},backend:r,attrs:{shape:[1,n.inChannels,n.outChannels]}});h.push(p);let g=$R({a:d,b:p,backend:r,transposeA:!1,transposeB:!1,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o}),_=r.texData.get(g.dataId);b(_.isPacked,()=>`batchMatMul result is expected to be packed`),l.shape=f,_.shape=n.outShape,m=gR({inputs:{x:g},backend:r}),m.shape=n.outShape,h.push(g)}else{let c=n.outHeight*n.outWidth,l=Z({inputs:{x:e},backend:r,attrs:{shape:p?[n.batchSize,c,n.inChannels]:[n.batchSize,n.inChannels,c]}}),u=Z({inputs:{x:t},backend:r,attrs:{shape:[1,n.inChannels,n.outChannels]}}),d=$R({a:p?l:u,b:p?u:l,transposeA:!p,transposeB:!1,backend:r,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o});m=Z({inputs:{x:d},backend:r,attrs:{shape:n.outShape}}),h.push(l),h.push(u),h.push(d)}for(let e of h)r.disposeIntermediateTensorInfo(e);return m}function UB({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let{filterWidth:c,filterHeight:l,inChannels:u,outWidth:d,outHeight:f,dataFormat:p}=n,m=p===`channelsLast`,h=c*l*u,g=f*d,_=[n.batchSize,h,g],v=[];if(a!=null){let e=VB(a.shape,m);e!=null&&(a=Z({inputs:{x:a},backend:r,attrs:{shape:e}}),v.push(a))}if(i!=null){let e=VB(i.shape,m);e!=null&&(i=Z({inputs:{x:i},backend:r,attrs:{shape:e}}),v.push(i))}let y=Z({inputs:{x:t},backend:r,attrs:{shape:[1,h,C(t.shape)/h]}});v.push(y);let b=new BB(_,n),x=[e.shape,[n.padInfo.top,n.padInfo.left],[n.strideHeight,n.strideWidth],[n.dilationHeight,n.dilationWidth],[n.inChannels],[n.filterWidth*n.inChannels],[n.outWidth]],S=r.runWebGLProgram(b,[e],`float32`,x),w=Z({inputs:{x:S},backend:r,attrs:{shape:_}});v.push(S),v.push(w);let T=i!=null,E=a!=null,D=s===`leakyrelu`,ee=s?jR(s,!0):null,te=new MR(m?w.shape:y.shape,m?y.shape:w.shape,m?[n.batchSize,g,n.outChannels]:[n.batchSize,n.outChannels,g],!0,!1,T,ee,E,D),O=m?[w,y]:[y,w];if(i&&O.push(i),E&&O.push(a),D){let e=r.makeTensorInfo([],`float32`,oi(o,`float32`));O.push(e),v.push(e)}let ne=r.runWebGLProgram(te,O,`float32`),re=Z({inputs:{x:ne},backend:r,attrs:{shape:n.outShape}});v.push(ne);for(let e of v)r.disposeIntermediateTensorInfo(e);return re}function WB(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dataFormat:c,dilations:l,dimRoundingMode:u}=r,d=gs(c),f=ns(i.shape,a.shape,o,l,s,u,!1,d),p;if(f.filterHeight===1&&f.filterWidth===1&&f.dilationHeight===1&&f.dilationWidth===1&&f.strideHeight===1&&f.strideWidth===1&&(f.padInfo.type===`SAME`||f.padInfo.type===`VALID`))p=HB({x:i,filter:a,convInfo:f,backend:n});else if(f.strideWidth<=2&&d===`channelsLast`&&A().getBool(`WEBGL_EXP_CONV`)){let e=new zB(f),t=[[f.padInfo.top,f.padInfo.left],[f.strideHeight,f.strideWidth],[f.dilationHeight,f.dilationWidth],[f.inHeight,f.inWidth]];p=n.runWebGLProgram(e,[i,a],`float32`,t)}else if(A().getBool(`WEBGL_CONV_IM2COL`))p=UB({x:i,filter:a,convInfo:f,backend:n});else{let e=new LB(f);p=n.runWebGLProgram(e,[i,a],`float32`)}let m=Z({inputs:{x:p},backend:n,attrs:{shape:f.outShape}});return n.disposeIntermediateTensorInfo(p),m}var GB={kernelName:lt,backendName:`webgl`,kernelFunc:WB},KB=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideHeight,n=e.strideWidth,r=e.padInfo.top,i=e.padInfo.left,a=e.dataFormat===`channelsLast`;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${r};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${n} - ${i};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              ${a?`float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);`:`float dyValue = getDy(b, d2, yR, yC);
              float xValue = getX(b, d1, xR, xC);
              dotProd += (xValue * dyValue);`}
            }
          }
        }
        setOutput(dotProd);
      }
    `}},qB=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=e.dataFormat===`channelsLast`,o=t-1-e.padInfo.top,s=n-1-e.padInfo.left,c=a?1:2,l=a?2:3,u=a?3:1;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${s});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${u}];

        ivec2 dyCorner = ivec2(coords[${c}], coords[${l}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${n}; wC++) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${n} - 1 - wC;

            for (int d2 = 0; d2 < ${e.outChannels}; d2++) {

              if (${a}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}},JB=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideDepth,n=e.strideHeight,r=e.strideWidth,i=e.padInfo.front,a=e.padInfo.top,o=e.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yF = 0; yF < ${e.outDepth}; yF++) {
            int xF = wF + yF * ${t} - ${i};

            if (xF < 0 || xF >= ${e.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${e.outHeight}; yR++) {
              int xR = wR + yR * ${n} - ${a};

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${e.outWidth}; yC++) {
                int xC = wC + yC * ${r} - ${o};

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}},YB=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterDepth,n=e.filterHeight,r=e.filterWidth,i=e.strideDepth,a=e.strideHeight,o=e.strideWidth,s=t-1-e.padInfo.front,c=n-1-e.padInfo.top,l=r-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${s}, ${c}, ${l});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${t}; wF++) {
          float dyF = float(dyFCorner + wF) / ${i}.0;

          if (dyF < 0.0 || dyF >= ${e.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${t} - 1 - wF;

          for (int wR = 0; wR < ${n}; wR++) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${n} - 1 - wR;

            for (int wC = 0; wC < ${r}; wC++) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${r} - 1 - wC;

              for (int d2 = 0; d2 < ${e.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function XB(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,filterShape:u}=r,d=gs(c),f=new KB(ns(i.shape,u,o,1,s,l,!1,d));return n.runWebGLProgram(f,[i,a],`float32`)}var ZB={kernelName:ut,backendName:`webgl`,kernelFunc:XB},QB=class{constructor(e){this.variableNames=[`dy`,`W`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`strides`,type:`vec2`}],this.outputShape=e.inShape,this.enableShapeUniforms=eI(this.outputShape.length);let t=e.filterHeight,n=e.filterWidth,r=t-1-e.padInfo.top,i=n-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${r}, ${i});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];

        ivec2 dyCorner = ivec2(coords[1], coords[2]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        vec4 result = vec4(0.);
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / strides[0];
          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);
          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${n}; wC++) {
            int wCPerm = ${n} - 1 - wC;

            float dyC = float(dyCCorner + wC) / strides[1];
            bool idyCVal = (dyC >= 0.0) && (dyC < ${e.outWidth}.0)
              && (fract(dyC) == 0.0);
            int idyC = int(dyC);

            float dyC2 = float(dyCCorner + wC + 1) / strides[1];
            bool idyCVal2 = (dyC2 >= 0.0) && (dyC2 < ${e.outWidth}.0)
              && (fract(dyC2) == 0.0);
            int idyC2 = int(dyC2);

            if (idyCVal && idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec4 dySample2 = (idyC / 2 == idyC2 / 2) ?
                  dySample : getDy(batch, idyR, idyC2, d2);

                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));

                dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample2.xy : dySample2.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC2, d2);
                vec2 dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            }
          }
        }
        setOutput(result);
      }
    `}};function $B(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{inputShape:o,strides:s,pad:c,dataFormat:l,dimRoundingMode:u}=r,d=gs(l),f=ns(o,a.shape,s,1,c,u,!1,d);if(A().getBool(`WEBGL_PACK_CONV2DTRANSPOSE`)&&d===`channelsLast`){let e=[[f.strideHeight,f.strideWidth]],t=new QB(f);return n.runWebGLProgram(t,[i,a],`float32`,e)}{let e=new qB(f);return n.runWebGLProgram(e,[i,a],`float32`)}}var eV={kernelName:dt,backendName:`webgl`,kernelFunc:$B};function tV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r,l=new RB(rs(i.shape,a.shape,o,c,s));return n.runWebGLProgram(l,[i,a],`float32`)}var nV={kernelName:ft,backendName:`webgl`,kernelFunc:tV};function rV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r,l=new JB(rs(i.shape,c,o,1,s));return n.runWebGLProgram(l,[i,a],`float32`)}var iV={kernelName:pt,backendName:`webgl`,kernelFunc:rV};function aV(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{pad:o,strides:s,inputShape:c}=r,l=new YB(rs(c,a.shape,s,1,o));return n.runWebGLProgram(l,[i,a],`float32`)}var oV={kernelName:mt,backendName:`webgl`,kernelFunc:aV},sV={kernelName:`Cos`,backendName:`webgl`,kernelFunc:kR({opSnippet:OR+`
  return cos(x);
`,packedOpSnippet:`
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${mR}
  return result;
`})},cV={kernelName:ht,backendName:`webgl`,kernelFunc:kR({opSnippet:`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`})},lV=class{constructor(e,t,n,r,i){this.variableNames=[`Image`,`Boxes`,`BoxInd`],this.outputShape=[];let[a,o,s,c]=e,[l]=t,[u,d]=n;this.outputShape=[l,u,d,c];let f=+(r===`bilinear`),[p,m]=[`${o-1}.0`,`${s-1}.0`],[h,g,_]=u>1?[`${(o-1)/(u-1)}`,`(y2-y1) * height_ratio`,`y1*${p} + float(y)*(height_scale)`]:[`0.0`,`0.0`,`0.5 * (y1+y2) * ${p}`],[v,y,b]=d>1?[`${(s-1)/(d-1)}`,`(x2-x1) * width_ratio`,`x1*${m} + float(x)*(width_scale)`]:[`0.0`,`0.0`,`0.5 * (x1+x2) * ${m}`];this.userCode=`
      const float height_ratio = float(${h});
      const float width_ratio = float(${v});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${a}) {
          return;
        }

        float height_scale = ${g};
        float width_scale = ${y};

        float in_y = ${_};
        if( in_y < 0.0 || in_y > ${p} ) {
          setOutput(float(${i}));
          return;
        }
        float in_x = ${b};
        if( in_x < 0.0 || in_x > ${m} ) {
          setOutput(float(${i}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${f} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}},uV={kernelName:vt,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{image:i,boxes:a,boxInd:o}=t,{cropSize:s,method:c,extrapolationValue:l}=r,u=new lV(i.shape,a.shape,s,c,l);return n.runWebGLProgram(u,[i,a,o],`float32`)}},dV;(function(e){e.Prod=`*`,e.Sum=`+`})(dV||={});var fV=class{constructor(e,t,n,r){this.op=e,this.outputShape=t,this.variableNames=[`x`],this.customUniforms=[{name:`index`,type:`float`}];let i=this.outputShape.length,a=this.op===dV.Prod?`1.0`:`0.0`,o=n?a:`getX(${pV(i,`coords`,this.op)})`,s=this.outputShape[this.outputShape.length-1],c=``,l=``;n?(c=r?`end != ${s-1}`:`end != 0`,l=r?`end + 1`:`end - 1`):(c=r?`end + pow2 < ${s}`:`end >= pow2`,l=r?`end + pow2`:`end - pow2`),this.userCode=`
      void main() {
        ${GF(i)} coords = getOutputCoords();
        int end = ${mV(i,`coords`,this.op)};
        float val = ${o};
        int pow2 = int(pow(2.0, index));
        if (${c}) {
          int idx = ${l};
          ${mV(i,`coords`,this.op)} = idx;
          val ${this.op}= getX(${pV(i,`coords`,this.op)});
        }
        setOutput(val);
      }
    `}};function pV(e,t,n){if(e===1)return`${t}`;if(e===2)return`${t}.x, ${t}.y`;if(e===3)return`${t}.x, ${t}.y, ${t}.z`;if(e===4)return`${t}.x, ${t}.y, ${t}.z, ${t}.w`;throw Error(`Cumulative ${n} for rank ${e} is not yet supported`)}function mV(e,t,n){if(e===1)return`${t}`;if(e===2)return`${t}.y`;if(e===3)return`${t}.z`;if(e===4)return`${t}.w`;throw Error(`Cumulative ${n} for rank ${e} is not yet supported`)}function hV(e,t,n,r,i,a){let o=t.shape.length,s=ol([r],o),c=t;s!=null&&(c=ZR({inputs:{x:t},backend:n,attrs:{perm:s}}));let l=cl(1,o)[0];if(l!==o-1)throw Error(`WebGL cumprod shader expects an inner-most axis=${t.shape.length-1} but got axis=${r}`);let u=c.shape[l],d=gR({inputs:{x:c},backend:n});for(let t=0;t<=Math.ceil(Math.log2(u))-1;t++){let r=new fV(e,c.shape,!1,a),i=[[t]],o=d;d=n.runWebGLProgram(r,[d],d.dtype,i),n.disposeIntermediateTensorInfo(o)}if(i){let t=new fV(e,c.shape,i,a),r=d;d=n.runWebGLProgram(t,[d],d.dtype),n.disposeIntermediateTensorInfo(r)}if(s!=null){let e=sl(s),t=ZR({inputs:{x:d},backend:n,attrs:{perm:e}});return n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(c),t}return d}function gV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;return hV(dV.Prod,i,n,a,o,s)}var _V={kernelName:gt,backendName:`webgl`,kernelFunc:gV};function vV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;return hV(dV.Sum,i,n,a,o,s)}var yV={kernelName:_t,backendName:`webgl`,kernelFunc:vV};function bV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o,binaryOutput:s}=r;if(i.shape.length===1){let e=NI(n.readSync(i.dataId),n.readSync(a.dataId),a.dtype,a.shape,o);return n.makeTensorInfo([o],a.dtype,e)}if(i.shape.length===2){let e=PI(n.bufferSync(i),n.bufferSync(a),o,s);return n.makeTensorInfo(e.shape,a.dtype,e.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${i.shape.length}.`)}var xV={kernelName:yt,backendName:`webgl`,kernelFunc:bV},SV=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=[],this.outputShape=e,this.blockSize=t,this.dataFormat=n,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${t};
      int offset_h = imod(h, ${t});
      int in_w = w / ${t};
      int offset_w = imod(w, ${t});
      int offset_d = (offset_h * ${t} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return this.dataFormat===`NHWC`?`coords[1]`:`coords[2]`}getWidthCoordString(){return this.dataFormat===`NHWC`?`coords[2]`:`coords[3]`}getDepthCoordString(){return this.dataFormat===`NHWC`?`coords[3]`:`coords[1]`}getOutputDepthSize(){return this.dataFormat===`NHWC`?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return this.dataFormat===`NHWC`?`getX(b, in_h, in_w, in_d)`:`getX(b, in_d, in_h, in_w)`}};function CV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockSize:a,dataFormat:o}=r,s=i.shape[0],c=o===`NHWC`?i.shape[1]:i.shape[2],l=o===`NHWC`?i.shape[2]:i.shape[3],u=o===`NHWC`?i.shape[3]:i.shape[1],d=c*a,f=l*a,p=u/(a*a),m=new SV(o===`NHWC`?[s,d,f,p]:[s,p,d,f],a,o);return n.runWebGLProgram(m,[i],i.dtype)}var wV={kernelName:bt,backendName:`webgl`,kernelFunc:CV},TV=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=eI(this.outputShape.length);let a=e.filterHeight,o=e.filterWidth,s=e.outChannels/e.inChannels,c=``,l=``;n&&(c=r?`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`
          float activation(float x) {
            ${n}
          }
        `,l=`result = activation(result);`);let u=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${c}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${s};
        int q = d2 - d1 * ${s};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${a}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${o}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${u}
        ${l}
        setOutput(result);
      }
    `}},EV=class{constructor(e,t=!1,n=null,r=!1,i=!1){this.variableNames=[`x`,`W`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`pads`,type:`ivec2`},{name:`strides`,type:`ivec2`},{name:`dilations`,type:`ivec2`},{name:`inDims`,type:`ivec2`}],this.outputShape=e.outShape,this.enableShapeUniforms=eI(this.outputShape.length);let a=e.outChannels/e.inChannels,o=e.padInfo.left,s=e.strideWidth,c=e.dilationWidth,l=e.filterHeight,u=e.filterWidth,d=u,f=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let e=0;e<u;e++)f+=`
          vec4 xTexelC${e*2};
          int xTexelC${e*2}Ready;
          vec4 xTexelC${e*2+1};
          int xTexelC${e*2+1}Ready;
          vec4 xC${e};`;f+=`
    for (int r = 0; r < ${l}; r++) {
      `;for(let e=0;e<u;e++)f+=`
          xTexelC${e*2} = vec4(0.0);
          xTexelC${e*2}Ready = 0;
          xTexelC${e*2+1} = vec4(0.0);
          xTexelC${e*2+1}Ready = 0;
          xC${e} = vec4(0.0);`;f+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let e=0;e<(d+1)/2;e++){let t=e*2;if(f+=`
          xC = xCCorner + ${t*c};
          `,s===1){if(t<u&&(o%2==1?(f+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }
              `,f+=c===1&&t>0?`
                xC${t} = vec4(xTexelC${t-2}.zw, xTexelC${t}.xy);
                `:`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${t} = vec4(previous.zw, xTexelC${t}.xy);
                  } else {
                    xC${t} = vec4(0.0, 0.0, xTexelC${t}.xy);
                  }
                  `):f+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                xC${t} = xTexelC${t};
                `,t+1<u)){let e=o%2==0?_(c):c;c%2==0&&o%2==1||c%2!=0&&o%2!=1?(f+=`
                  xCOffset = xC + imod(pads[1], 2) + ${e};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                    xTexelC${t+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${t+1}.zw = vec2(0.0);
                    }
                    xTexelC${t+1}Ready = 1;
                  }
                  `,f+=c>1?`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${t+1} = vec4(previous.zw, xTexelC${t+1}.xy);
                    } else {
                     xC${t+1} = vec4(0.0, 0.0, xTexelC${t+1}.xy);
                    }
                    `:`
                    xC${t+1} = vec4(xTexelC${t}.zw, xTexelC${t+1}.xy);
                    `):f+=e===1?`
                    xC${t+1} = xTexelC${t};
                    `:`
                    xCOffset = xC + ${e};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                      xTexelC${t+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${t+1}.zw = vec2(0.0);
                      }
                      xTexelC${t+1}Ready = 1;
                    }

                    xC${t+1} = xTexelC${t+1};
                    `}}else t<u&&(o%2==1?(f+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${t+1}Ready == 0) {
                  xTexelC${t+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${t+1}.zw = vec2(0.0);
                  }
                  xTexelC${t+1}Ready = 1;
                }

                xC${t} = vec4(xTexelC${t}.zw, xTexelC${t+1}.zw);
              `,t+1<u&&(f+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${t+1} = vec4(xTexelC${t+1}.xy, final.xy);
                `)):(f+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                  xTexelC${t+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t+1}.zw = vec2(0.);
                  }
                  xTexelC${t+1}Ready = 1;
                }

                xC${t} = vec4(
                  xTexelC${t}.xy, xTexelC${t+1}.xy);
              `,t+1<u&&(f+=`
                  xC${t+1} = vec4(xTexelC${t}.zw, xTexelC${t+1}.zw);
                `)));t<u&&(f+=`
            wTexel = getW(r, ${t}, d1, q);
            dotProd += xC${t} * vec4(wTexel.xz, wTexel.xz);
          `,t+1<u&&(f+=`
              wTexel = getW(r, ${t+1}, d1, q);
              dotProd += xC${t+1} * vec4(wTexel.xz, wTexel.xz);
            `))}f+=`
    }
  `,f+=`
      }
    `;let p=``,m=``;n&&(p=r?`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${n}
        }`:i?`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${n}
        }`:`vec4 activation(vec4 x) {
          ${n}
        }`,m=`result = activation(result);`);let h=t?`result += getBiasAtOutCoords();`:``;t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),i&&this.variableNames.push(`leakyreluAlpha`),this.userCode=`
      ${p}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${a};
        int q = d2 - d1 * ${a};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${f}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${h}
        ${m}
        setOutput(result);
      }
    `}};function DV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c,dimRoundingMode:l}=r,u=c;u??=[1,1],b(ms(o,u),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${o} and dilations '${u}'`);let d=ns(i.shape,a.shape,o,u,s,l,!0),f;f=A().getBool(`WEBGL_PACK_DEPTHWISECONV`)&&d.strideWidth<=2&&d.outChannels/d.inChannels===1?new EV(d):new TV(d);let p=[[d.padInfo.top,d.padInfo.left],[d.strideHeight,d.strideWidth],[d.dilationHeight,d.dilationWidth],[d.inHeight,d.inWidth]];return n.runWebGLProgram(f,[i,a],`float32`,p)}var OV={kernelName:xt,backendName:`webgl`,kernelFunc:DV},kV=class{constructor(e){this.variableNames=[`x`,`dy`],this.outputShape=e.filterShape;let t=e.strideHeight,n=e.strideWidth,r=e.padInfo.top,i=e.padInfo.left,a=e.outChannels/e.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${a} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${r};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${n} - ${i};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}},AV=class{constructor(e){this.variableNames=[`dy`,`W`],this.outputShape=e.inShape;let t=e.filterHeight,n=e.filterWidth,r=e.strideHeight,i=e.strideWidth,a=t-1-e.padInfo.top,o=n-1-e.padInfo.left,s=e.outChannels/e.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${n}; wC++) {
            float dyC = float(dyCCorner + wC) / ${i}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${n} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${s}; dm++) {
              int d2 = d1 * ${s} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function jV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,filterShape:u}=r,d=new kV(ns(i.shape,u,o,s,c,l,!0));return n.runWebGLProgram(d,[i,a],`float32`)}var MV={kernelName:St,backendName:`webgl`,kernelFunc:jV};function NV(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,inputShape:u}=r,d=new AV(ns(u,a.shape,o,s,c,l,!0));return n.runWebGLProgram(d,[i,a],`float32`)}var PV={kernelName:Ct,backendName:`webgl`,kernelFunc:NV},FV=class{constructor(e){this.variableNames=[`X`],this.outputShape=[e,e],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}};function IV(e){let{inputs:t,backend:n}=e,{x:r}=t,i=[...r.shape,...r.shape],a=C(r.shape),o=Z({inputs:{x:r},backend:n,attrs:{shape:[a]}}),s=new FV(a),c=n.runWebGLProgram(s,[o],o.dtype),l=Z({inputs:{x:c},backend:n,attrs:{shape:i}});return n.disposeIntermediateTensorInfo(o),n.disposeIntermediateTensorInfo(c),l}var LV={kernelName:wt,backendName:`webgl`,kernelFunc:IV},RV=class{constructor(e){this.variableNames=[`x`,`W`],this.outputShape=e.outShape;let{inHeight:t,inWidth:n,padInfo:r,strideHeight:i,strideWidth:a,filterHeight:o,filterWidth:s,dilationHeight:c,dilationWidth:l}=e,{top:u,left:d}=r;this.userCode=`
      const ivec2 strides = ivec2(${i}, ${a});
      const ivec2 pads = ivec2(${u}, ${d});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${o}; h++) {
          int hIn = hBeg + h * ${c};

          if (hIn >= 0 && hIn < ${t}) {
            for (int w = 0; w < ${s}; w++) {
              int wIn = wBeg + w * ${l};

              if (wIn >= 0 && wIn < ${n}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}};function zV(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r,l=$o(i.shape,a.shape,o,s,`NHWC`,c),u,d=new RV(l);u=n.runWebGLProgram(d,[i,a],`float32`);let f=Z({inputs:{x:u},backend:n,attrs:{shape:l.outShape}});return n.disposeIntermediateTensorInfo(u),f}var BV={kernelName:Tt,backendName:`webgl`,kernelFunc:zV};function VV(e){let{inputs:t,backend:n,attrs:r}=e,{equation:i}=r,a=t,{allDims:o,summedDims:s,idDims:c}=Qh(i,a.length);eg(o.length,c,a);let{path:l,steps:u}=tg(s,c),d=u.length,f=null,p=o.length,m=[];for(let e=0;e<d;++e){for(let t of u[e]){let{permutationIndices:e,expandDims:r}=$h(p,c[t]),i;ng(e)?i=a[t]:(i=ZR({inputs:{x:a[t]},backend:n,attrs:{perm:e}}),m.push(i));let o=i.shape.slice();for(let e=0;e<r.length;++e)o.splice(r[e],0,1);w(i.shape,o)||(i=Z({inputs:{x:i},backend:n,attrs:{shape:o}}),m.push(i)),f===null?f=i:(f=IR({inputs:{a:i,b:f},backend:n}),m.push(f))}e<d-1&&(l[e]>=0&&(f=YR({inputs:{x:f},backend:n,attrs:{axis:l[e]-(o.length-p),keepDims:!1}}),m.push(f)),p--)}for(let e of m)e!==f&&n.disposeIntermediateTensorInfo(e);return f}var HV={kernelName:At,backendName:`webgl`,kernelFunc:VV},UV={kernelName:`Elu`,backendName:`webgl`,kernelFunc:kR({opSnippet:`return (x >= 0.0) ? x : (exp(x) - 1.0);`,packedOpSnippet:`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`})},WV=`return (b >= 0.0) ? a : a * (b + 1.0);`,GV=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,KV={kernelName:jt,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n}=e,{dy:r,y:i}=t,a=A().getBool(`WEBGL_PACK_BINARY_OPERATIONS`)?new hR(GV,r.shape,i.shape):new pR(WV,r.shape,i.shape);return n.runWebGLProgram(a,[r,i],r.dtype)}},qV={kernelName:Mt,backendName:`webgl`,kernelFunc:AR({opSnippet:`return float(a == b);`,packedOpSnippet:`
  return vec4(equal(a, b));
`,dtype:`bool`,cpuKernelImpl:zI})},JV={kernelName:`Erf`,backendName:`webgl`,kernelFunc:kR({opSnippet:`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${Ph};
  float a1 = ${Fh};
  float a2 = ${Ih};
  float a3 = ${Lh};
  float a4 = ${Rh};
  float a5 = ${zh};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`})},YV=kR({opSnippet:OR+`
  return exp(x);
`,packedOpSnippet:`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:BI,dtype:`float32`}),XV={kernelName:`Exp`,backendName:`webgl`,kernelFunc:YV};function ZV(e){let{inputs:t,attrs:n,backend:r}=e,{dim:i}=n,{input:a}=t,o=a.shape.length,s=a.shape.slice(),c=i;return i<0&&(b(-(o+1)<=i,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+i+1),s.splice(c,0,1),Z({inputs:{x:a},backend:r,attrs:{shape:s}})}var QV={kernelName:Nt,backendName:`webgl`,kernelFunc:ZV},$V=`return exp(x) - 1.0;`,eH={kernelName:Pt,backendName:`webgl`,kernelFunc:kR({opSnippet:$V,packedOpSnippet:$V,cpuKernelImpl:VI})},tH=class{constructor(e,t,n){this.variableNames=[`real`,`imag`];let r=t[1];this.outputShape=t;let i=n?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,a=n?`${r}.0`:`1.0`,o;if(e===`real`)o=`return real * expR - imag * expI;`;else if(e===`imag`)o=`return real * expI + imag * expR;`;else throw Error(`FFT component must be either "real" or "imag", got ${e}.`);this.userCode=`
      const float exponentMultiplier = ${i};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${o}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${r});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${r}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${a};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}};function nH(e,t,n){let r=n.texData.get(e.dataId),i=C(e.shape),a=e.shape[e.shape.length-1],o=i/a,s=Z({inputs:{x:e},backend:n,attrs:{shape:[o,a]}}),c=s.shape,l=new tH(`real`,c,t),u=new tH(`imag`,c,t),d=[{dataId:r.complexTensorInfos.real.dataId,dtype:r.complexTensorInfos.real.dtype,shape:c},{dataId:r.complexTensorInfos.imag.dataId,dtype:r.complexTensorInfos.imag.dtype,shape:c}],f=n.runWebGLProgram(l,d,`float32`),p=n.runWebGLProgram(u,d,`float32`),m=vR({inputs:{real:f,imag:p},backend:n});n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p);let h=Z({inputs:{x:m},backend:n,attrs:{shape:e.shape}});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(m),h}function rH(e){let{inputs:t,backend:n}=e,{input:r}=t;return nH(r,!1,n)}var iH={kernelName:`FFT`,backendName:`webgl`,kernelFunc:rH},aH=class{constructor(e,t){this.outputShape=[],this.customUniforms=[{name:`value`,type:`float`}],this.variableNames=[`x`],this.outputShape=e,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}};function oH(e){let{backend:t,attrs:n}=e,{shape:r,value:i}=n,{dtype:a}=n;if(a||=pe(i),a===`string`){let e=ie(a,C(r));return e.fill(i),t.makeTensorInfo(r,a,e)}{let e=new aH(r,i),n=[[i]];return t.runWebGLProgram(e,[],a,n)}}var sH={kernelName:Ft,backendName:`webgl`,kernelFunc:oH},cH=class{constructor(e){this.variableNames=[`Image`],this.outputShape=[];let t=e[2];this.outputShape=e,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${t} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${t}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}},lH={kernelName:It,backendName:`webgl`,kernelFunc:({inputs:e,backend:t})=>{let{image:n}=e,r=t,i=new cH(n.shape);return r.runWebGLProgram(i,[n],n.dtype)}},uH=`return floor(x);`,dH={kernelName:Lt,backendName:`webgl`,kernelFunc:kR({opSnippet:uH,packedOpSnippet:uH,cpuKernelImpl:HI})},fH={kernelName:Rt,backendName:`webgl`,kernelFunc:AR({opSnippet:`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,packedOpSnippet:`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,dtype:`int32`})},pH=class{constructor(e){this.variableNames=[`A`];let t=YP(),[n,r]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${r}.0, ${n}.0);

        vec4 values = ${t.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}},mH=class{constructor(e){this.variableNames=[`A`],this.packedInputs=!1,this.packedOutput=!0;let t=YP(),[n,r]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${r}.0, ${n}.0);
            vec4 values = ${t.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${t.output} = result;
      }
    `}},hH={kernelName:Er,backendName:`webgl`,kernelFunc:vH},gH,_H=A().getBool(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`);function vH(e){let{inputs:t,backend:n,attrs:r}=e,{pixels:i}=t,{numChannels:a}=r,o=typeof HTMLVideoElement<`u`&&i instanceof HTMLVideoElement,s=typeof HTMLImageElement<`u`&&i instanceof HTMLImageElement,[c,l]=o?[i.videoWidth,i.videoHeight]:[i.width,i.height],u=[l,c],d=[l,c,a];if(s||o){let e=A().getBool(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`);(gH==null||e!==_H)&&(_H=e,gH=document.createElement(`canvas`).getContext(`2d`,{willReadFrequently:_H})),gH.canvas.width=c,gH.canvas.height=l,gH.drawImage(i,0,0,c,l),i=gH.canvas}let f=n.makeTensorInfo(u,`int32`);n.texData.get(f.dataId).usage=qN.PIXELS,n.gpgpu.uploadPixelDataToTexture(n.getTexture(f.dataId),i);let p=A().getBool(`WEBGL_PACK`)?new mH(d):new pH(d),m=n.runWebGLProgram(p,[f],`int32`);return n.disposeData(f.dataId),m}function yH(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=gs(u),g=ns(i.shape,a.shape,c,d,l,f,!1,h),_,v=[],y=o!=null,b=s!=null,x=p===`leakyrelu`,S=()=>{let e=[i,a],t=(e,t)=>{if(t===`NCHW`&&e.shape.length===1&&e.shape[0]!==1){let t=Z({inputs:{x:e},backend:n,attrs:{shape:[e.shape[0],1,1]}});return v.push(t),t}return e};if(y&&e.push(t(o,u)),b&&e.push(t(s,u)),x){let t=n.makeTensorInfo([],`float32`,oi(m,`float32`));e.push(t),v.push(t)}return e};if(g.filterHeight===1&&g.filterWidth===1&&g.dilationHeight===1&&g.dilationWidth===1&&g.strideHeight===1&&g.strideWidth===1&&(g.padInfo.type===`SAME`||g.padInfo.type===`VALID`))_=HB({x:i,filter:a,convInfo:g,backend:n,bias:o,activation:p,preluActivationWeights:s,leakyreluAlpha:m});else if(g.strideWidth<=2&&h===`channelsLast`&&A().getBool(`WEBGL_EXP_CONV`)){let e=new zB(g,y,p?jR(p,!0):null,b,x),t=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],r=S();_=n.runWebGLProgram(e,r,`float32`,t)}else if(A().getBool(`WEBGL_CONV_IM2COL`))_=UB({x:i,filter:a,convInfo:g,backend:n,bias:o,activation:p,preluActivationWeights:s,leakyreluAlpha:m});else{let e=new LB(g,y,p?jR(p,!1):null,b,x),t=S();_=n.runWebGLProgram(e,t,`float32`)}let C=Z({inputs:{x:_},backend:n,attrs:{shape:g.outShape}});return v.push(_),v.forEach(e=>n.disposeIntermediateTensorInfo(e)),C}var bH={kernelName:kr,backendName:`webgl`,kernelFunc:yH};function xH(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dilations:u,dimRoundingMode:d,activation:f,leakyreluAlpha:p}=r,m=[],h=u;h??=[1,1],b(ms(c,h),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${c} and dilations '${h}'`);let g=ns(i.shape,a.shape,c,h,l,d,!0),_=A().getBool(`WEBGL_PACK_DEPTHWISECONV`)&&g.strideWidth<=2&&g.outChannels/g.inChannels===1,v=f?jR(f,_):null,y=[i,a],x=o!=null,S=s!=null,C=f===`leakyrelu`;if(x&&y.push(o),S&&y.push(s),C){let e=n.makeTensorInfo([],`float32`,oi(p,`float32`));y.push(e),m.push(e)}let w;w=_?new EV(g,x,v,S,C):new TV(g,x,v,S,C);let T=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],E=n.runWebGLProgram(w,y,`float32`,T);return m.forEach(e=>n.disposeIntermediateTensorInfo(e)),E}var SH={kernelName:Ar,backendName:`webgl`,kernelFunc:xH},CH=class{constructor(e,t,n,r){this.sliceDim=e,this.strides=t,this.paramsShape=r,this.variableNames=[`x`,`indices`],this.outputShape=n;let i=GF(n.length),a=`
    int index;`;for(let e=0;e<this.sliceDim;e++)a+=`
          index = round(getIndices(coords[0], ${e}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[e]};
          flattenIndex += index * ${this.strides[e]};`;this.userCode=`
         void main() {
          ${i} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${a}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}};function wH(e){let{inputs:t,backend:n}=e,{params:r,indices:i}=t,a=i.shape,o=a[a.length-1],s=C(r.shape),[c,l,u,d]=qm(r,i),f=Z({inputs:{x:i},backend:n,attrs:{shape:[l,o]}}),p=Z({inputs:{x:r},backend:n,attrs:{shape:[C(r.shape)/u,u]}});if(n.shouldExecuteOnCPU([r,i])||r.dtype===`string`){let e=UI(n.readSync(i.dataId),n.bufferSync(r),r.dtype,l,o,u,d,r.shape,s);return n.makeTensorInfo(c,r.dtype,e.values)}let m=new CH(o,d,[l,u],r.shape),h=n.runWebGLProgram(m,[p,f],p.dtype),g=Z({inputs:{x:h},backend:n,attrs:{shape:c}});return n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(h),g}var TH={kernelName:Vt,backendName:`webgl`,kernelFunc:wH},EH=class{constructor(e,t){this.variableNames=[`A`,`indices`],this.outputShape=t,this.rank=t.length;let n=GF(this.rank),r=DH(e,2);this.userCode=`
      void main() {
        ${n} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${e[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${r}));
      }
    `}};function DH(e,t){let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],r=[];for(let t=0;t<e.length;t++)t===2?r.push(`index`):r.push(`${n[t]}`);return r.join()}function OH(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,indices:a}=t,{axis:o,batchDims:s}=r,c=O(o,i.shape)[0];if(A().get(`DEBUG`)){let e=n.readSync(a.dataId),t=i.shape[c];for(let n=0;n<e.length;++n){let r=e[n];b(r<=t-1&&r>=0,()=>`GatherV2: the index value ${r} is not in [0, ${t-1}]`)}}let l=bg(i,a,c,s),u=C(a.shape),d=[],f=Z({inputs:{x:i},backend:n,attrs:{shape:[l.batchSize,l.outerSize,l.dimSize,l.sliceSize]}}),p=Z({inputs:{x:a},backend:n,attrs:{shape:[l.batchSize,u/l.batchSize]}});d.push(f),d.push(p);let m=[l.batchSize,l.outerSize,u/l.batchSize,l.sliceSize];if(n.shouldExecuteOnCPU([i,a])||i.dtype===`string`){let e=n.bufferSync(p),t=WI(n.bufferSync(f),e,m);return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.makeTensorInfo(l.outputShape,t.dtype,t.values)}let h=new EH(f.shape,m),g=n.runWebGLProgram(h,[f,p],f.dtype);d.push(g);let _=Z({inputs:{x:g},backend:n,attrs:{shape:l.outputShape}});return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),_}var kH={kernelName:Bt,backendName:`webgl`,kernelFunc:OH},AH={kernelName:Ht,backendName:`webgl`,kernelFunc:AR({opSnippet:`return float(a > b);`,packedOpSnippet:`
  return vec4(greaterThan(a, b));
`,cpuKernelImpl:GI,dtype:`bool`})},jH={kernelName:Ut,backendName:`webgl`,kernelFunc:AR({opSnippet:`return float(a >= b);`,packedOpSnippet:`
  return vec4(greaterThanEqual(a, b));
`,dtype:`bool`,cpuKernelImpl:KI})};function MH(e){let{inputs:t,backend:n}=e,{input:r}=t;return nH(r,!0,n)}var NH={kernelName:Gt,backendName:`webgl`,kernelFunc:MH},PH={kernelName:qt,backendName:`webgl`,kernelFunc:kR({opSnippet:`return float(!isnan(x) && !isinf(x));`,dtype:`bool`})},FH={kernelName:Jt,backendName:`webgl`,kernelFunc:kR({opSnippet:`return float(isinf(x));`,dtype:`bool`})},IH={kernelName:Yt,backendName:`webgl`,kernelFunc:kR({opSnippet:`return float(isnan(x));`,dtype:`bool`})},LH={kernelName:Zt,backendName:`webgl`,kernelFunc:AR({opSnippet:`return float(a < b);`,packedOpSnippet:`
  return vec4(lessThan(a, b));
`,cpuKernelImpl:qI,dtype:`bool`})},RH={kernelName:Qt,backendName:`webgl`,kernelFunc:AR({opSnippet:`return float(a <= b);`,packedOpSnippet:`
  return vec4(lessThanEqual(a, b));
`,cpuKernelImpl:JI,dtype:`bool`})};function zH(e){let{backend:t,attrs:n}=e,{start:r,stop:i,num:a}=n,o=YI(r,i,a);return t.makeTensorInfo([o.length],`float32`,o)}var BH={kernelName:$t,backendName:`webgl`,kernelFunc:zH},VH={kernelName:`Log`,backendName:`webgl`,kernelFunc:kR({opSnippet:OR+`
  return x < 0.0 ? 0./0. : log(x);
`,packedOpSnippet:`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,cpuKernelImpl:XI})},HH={kernelName:en,backendName:`webgl`,kernelFunc:kR({opSnippet:OR+`
  return log(1.0 + x);
`})},UH={kernelName:tn,backendName:`webgl`,kernelFunc:AR({opSnippet:`return float(a >= 1.0 && b >= 1.0);`,packedOpSnippet:`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,dtype:`bool`})},WH={kernelName:nn,backendName:`webgl`,kernelFunc:kR({opSnippet:`return float(!(x >= 1.0));`})},GH={kernelName:rn,backendName:`webgl`,kernelFunc:AR({opSnippet:`return float(a >= 1.0 || b >= 1.0);`,packedOpSnippet:`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,dtype:`bool`})},KH=class{constructor(e,t,n,r,i){this.variableNames=[`x`],this.outputShape=[];let a=t,o=e[3]-1;this.outputShape=e;let s,c=`float(${n}) + float(${r}) * sum`;s=i===.5?`inversesqrt(${c})`:i===1?`1.0/(${c})`:`exp(log(${c}) * float(-${i}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${a}; j <= ${a}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${o}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${s};
        setOutput(val);
      }
    `}},qH=class{constructor(e,t,n,r,i){this.variableNames=[`x`],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;let a=t,o=e[3]-1;this.outputShape=e;let s,c=`float(${n}) + float(${r}) * sum`;s=i===.5?`inversesqrt(${c})`:i===1?`1.0/(${c})`:`exp(log(${c}) * float(-${i}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${a};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${a}; j <= ${a}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${o}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${s};
        setOutput(result);
      }
    `}},JH={kernelName:`LRN`,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r,l=A().getBool(`WEBGL_PACK_NORMALIZATION`)?new qH(i.shape,a,o,s,c):new KH(i.shape,a,o,s,c);return n.runWebGLProgram(l,[i],i.dtype)}},YH=class{constructor(e,t,n,r,i){this.variableNames=[`inputImage`,`outputImage`,`dy`],this.outputShape=[],this.outputShape=e,this.depth=e[3],this.depthRadius=t,this.bias=n,this.alpha=r,this.beta=i,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${t})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${t} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${r}) * norm + float(${n});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${r})
                * float(${i})
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${i});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}},XH={kernelName:on,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r,d=new YH(i.shape,s,c,l,u);return n.runWebGLProgram(d,[i,a,o],i.dtype)}};function ZH(e,t,n,r){let i=C(t),a=C(e.shape)/i,o=Z({inputs:{x:e},attrs:{shape:[a,i]},backend:r}),s=UR(o,e.dtype,`max`,r),c=Z({inputs:{x:s},attrs:{shape:n},backend:r});return r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(s),c}function QH(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reductionIndices:a,keepDims:o}=r,s=i.shape.length,c=O(a,i.shape),l=c,u=ol(l,s),d=u!=null,f=n.shouldExecuteOnCPU([i]),p=i;if(d){if(f){let e=n.texData.get(p.dataId).values,t=Array(s);for(let e=0;e<t.length;e++)t[e]=i.shape[u[e]];let r=TL(e,i.shape,i.dtype,u,t);p=n.makeTensorInfo(t,i.dtype);let a=n.texData.get(p.dataId);a.values=r}else p=qR(i,u,n);l=cl(l.length,s)}al(`max`,l,s);let[m,h]=rl(p.shape,l),g=m;o&&(g=il(m,c));let _;if(f){let e=n.texData.get(p.dataId).values,t=ZI(e,C(h),g,i.dtype);_=n.makeTensorInfo(g,i.dtype);let r=n.texData.get(_.dataId);r.values=t}else _=ZH(p,h,g,n);return d&&n.disposeIntermediateTensorInfo(p),_}var $H={kernelName:`Max`,backendName:`webgl`,kernelFunc:QH},eU={kernelName:sn,backendName:`webgl`,kernelFunc:AR({opSnippet:fR+`
  return max(a, b);
`,packedOpSnippet:`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+mR+`
  return result;
`,cpuKernelImpl:QI})};function tU(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;JP(i,`maxPool`);let{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;b(ms(o,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let l=es(i.shape,a,o,1,s,c);if(l.filterWidth===1&&l.filterHeight===1&&w(l.inShape,l.outShape))return gR({inputs:{x:i},backend:n});let u=new jz(l,`max`,!1);return n.runWebGLProgram(u,[i],i.dtype)}var nU={kernelName:cn,backendName:`webgl`,kernelFunc:tU};function rU(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dataFormat:c,dimRoundingMode:l}=r,u=new Mz(ts(i.shape,a,o,[1,1,1],s,l,c),`max`,!1);return n.runWebGLProgram(u,[i],i.dtype)}var iU={kernelName:un,backendName:`webgl`,kernelFunc:rU},aU=class{constructor(e){this.variableNames=[`dy`,`maxPos`],this.outputShape=e.inShape;let t=e.strideHeight,n=e.strideWidth,r=e.dilationHeight,i=e.effectiveFilterHeight,a=e.effectiveFilterWidth,o=i-1-e.padInfo.top,s=a-1-e.padInfo.left,c=i*a-1;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${s});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${i};
          wR += ${r}) {
          float dyR = float(dyRCorner + wR) / ${t}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${a}; wC++) {
            float dyC = float(dyCCorner + wC) / ${n}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${c} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${a} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}},oU=class{constructor(e){this.variableNames=[`dy`,`maxPos`],this.outputShape=e.inShape;let t=e.strideDepth,n=e.strideHeight,r=e.strideWidth,i=e.dilationDepth,a=e.dilationHeight,o=e.dilationWidth,s=e.effectiveFilterDepth,c=e.effectiveFilterHeight,l=e.effectiveFilterWidth,u=s-1-e.padInfo.front,d=c-1-e.padInfo.top,f=l-1-e.padInfo.left,p=s*c*l-1;this.userCode=`
      const ivec3 pads = ivec3(${u}, ${d}, ${f});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${s};
           wD += ${i}) {
          float dyD = float(dyDCorner + wD) / ${t}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${c};
              wR += ${a}) {
            float dyR = float(dyRCorner + wR) / ${n}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${l};
                wC += ${o}) {
              float dyC = float(dyCCorner + wC) / ${r}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${p} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${c} * ${l} +
                  wR * ${l} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}};function sU(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a,{filterSize:s,strides:c,pad:l,dimRoundingMode:u}=r,d=ts(o.shape,s,c,[1,1,1],l,u),f=new Mz(d,`max`,!0),p=n.runWebGLProgram(f,[o],o.dtype),m=new oU(d),h=n.runWebGLProgram(m,[i,p],o.dtype);return n.disposeIntermediateTensorInfo(p),h}var cU={kernelName:dn,backendName:`webgl`,kernelFunc:sU};function lU(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a,output:o}=t,s=a;JP([a,o],`maxPoolGrad`);let{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=r,f=es(s.shape,c,l,1,u,d),p=new jz(f,`max`,!0),m=n.runWebGLProgram(p,[s],s.dtype),h=new aU(f),g=n.runWebGLProgram(h,[i,m],s.dtype);return n.disposeIntermediateTensorInfo(m),g}var uU={kernelName:ln,backendName:`webgl`,kernelFunc:lU};function dU(e,t,n,r){let i=new jz(n,`max`,!1),a=r.runWebGLProgram(i,[e],`float32`);return i=new jz(n,`max`,!0,!0,t),[a,r.runWebGLProgram(i,[e],`float32`)]}var fU={kernelName:fn,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{filterSize:i,strides:a,pad:o,includeBatchInIndex:s}=t,c=n;b(r.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${r.shape.length}.`);let l=[1,1];b(ms(a,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);let[u,d]=dU(r,s,es(r.shape,i,a,l,o),c);return[u,d]}};function pU(e,t,n,r){let i=C(t),a=C(e.shape)/i,o=Z({inputs:{x:e},attrs:{shape:[a,i]},backend:r}),s=UR(o,`float32`,`mean`,r),c=Z({inputs:{x:s},attrs:{shape:n},backend:r});return r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(s),c}var mU={kernelName:pn,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{keepDims:i,axis:a}=t,o=n,s=r.shape.length,c=O(a,r.shape),l=c,u=ol(l,s),d=u!=null,f=o.shouldExecuteOnCPU([r]),p=[],m=r;if(d){if(f){let e=o.texData.get(m.dataId).values,t=Array(s);for(let e=0;e<t.length;e++)t[e]=r.shape[u[e]];let n=TL(e,r.shape,r.dtype,u,t);m=o.makeTensorInfo(t,r.dtype);let i=o.texData.get(m.dataId);i.values=n}else m=qR(r,u,o);p.push(m),l=cl(l.length,s)}al(`sum`,l,s);let[h,g]=rl(m.shape,l),_=h;i&&(_=il(h,c));let v=pU(m,g,_,o);for(let e of p)o.disposeIntermediateTensorInfo(e);return v}};function hU(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=O(a,i.shape),l=c,u=ol(l,s),d=i;u!=null&&(d=ZR({inputs:{x:i},backend:n,attrs:{perm:u}}),l=cl(l.length,i.shape.length)),al(`min`,l,s);let[f,p]=rl(d.shape,l),m=C(p),h=Z({inputs:{x:d},backend:n,attrs:{shape:[-1,m]}}),g=UR(h,h.dtype,`min`,n),_;if(o){let e=il(f,c);_=Z({inputs:{x:g},backend:n,attrs:{shape:e}})}else _=Z({inputs:{x:g},backend:n,attrs:{shape:f}});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(g),u!=null&&n.disposeIntermediateTensorInfo(d),_}var gU={kernelName:`Min`,backendName:`webgl`,kernelFunc:hU},_U={kernelName:mn,backendName:`webgl`,kernelFunc:AR({opSnippet:fR+`
  return min(a, b);
`,packedOpSnippet:`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+mR+`
  return result;
`,cpuKernelImpl:$I})},vU=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=GF(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=[`coords[0]`,`coords[1]`,`coords[2]`,`coords[3]`].slice(0,r),c=n===`reflect`?0:1;if(r===1){this.userCode=`
        int start = ${a};
        int end = ${o};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${c};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${c};
          }
          setOutput(getX(outC - start));
        }
      `;return}this.userCode=`
      ${i} start = ${i}(${a});
      ${i} end = ${i}(${o});

      void main() {
        ${i} outC = getOutputCoords();
        for (int i = 0; i < ${r}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${c};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${c};
          }
        }
        ${i} coords = outC - start;
        setOutput(getX(${s}));
      }
    `}},yU=class{constructor(e,t,n){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=GF(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=OL(`rc`,r),c=OL(`source`,r),l=`${s[r-1]} < ${this.outputShape[r-1]}`,u=r===1?`source`:`vec2(${c.slice(-2).join()})`,d=n===`reflect`?0:1,f=``;if(r===1){let e=`
        ${i} source = rc;
        if (source < start) {
          source = start * 2 - source - ${d};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${d};
        }
        source -= start;
      `;f=`
        ${i} rc = outputLoc;
        ${e}
        result[0] = getChannel(getX(${c.join()}), ${u});
        ${s[r-1]} += 1;
        if(${l}) {
          ${e}
          result[1] = getChannel(getX(${c.join()}), ${u});
        }
      `}else{let e=`
        ${i} source = rc;
        ${i} lt = ${i}(lessThan(source, start));
        ${i} gte = ${i}(greaterThanEqual(source, end));
        ${i} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${d}) +
                gte * ((end - 1) * 2 - source + ${d});
        source -= start;
      `;f=`
        ${i} rc = outputLoc;
        ${e}
        result[0] = getChannel(getX(${c.join()}), ${u});
        ${s[r-1]} += 1;
        if(${l}) {
          ${e}
          result[1] = getChannel(getX(${c.join()}), ${u});
        }
        rc = outputLoc;
        ${s[r-2]} += 1;
        if(${s[r-2]} < ${this.outputShape[r-2]}) {
          ${e}
          result[2] = getChannel(getX(${c.join()}), ${u});
          ${s[r-1]} += 1;
          if(${l}) {
            ${e}
            result[3] = getChannel(getX(${c.join()}), ${u});
          }
        }
      `}this.userCode=`
      const ${i} start = ${i}(${a});
      const ${i} end = ${i}(${o});

      void main() {
        ${i} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${f}
        setOutput(result);
      }
    `}},bU={kernelName:hn,backendName:`webgl`,kernelFunc:({inputs:e,backend:t,attrs:n})=>{let{x:r}=e,{paddings:i,mode:a}=n,o=A().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new yU(r.shape,i,a):new vU(r.shape,i,a);return t.runWebGLProgram(o,[r],r.dtype)}},xU={kernelName:`Mod`,backendName:`webgl`,kernelFunc:AR({opSnippet:`if (b == 0.0) return NAN;
  return mod(a, b);`,packedOpSnippet:`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+mR+`
  return result;
`})},SU=class{constructor(e,t,n){this.variableNames=[`probs`],this.customUniforms=[{name:`seed`,type:`float`}],this.outputShape=[e,n],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${t-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${t-1}));
      }
    `}},CU=AR({opSnippet:`
if (a == b) {
  return 1.0;
};
return a / b;`,packedOpSnippet:`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,checkOutOfBounds:!0}),wU={kernelName:kt,backendName:`webgl`,kernelFunc:CU},TU=`return a - b;`,EU=AR({opSnippet:TU,packedOpSnippet:TU,supportsComplex:!0,cpuKernelImpl:SL}),DU={kernelName:`Sub`,backendName:`webgl`,kernelFunc:EU};function OU(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{dim:a}=r,o=O([a],i.shape),s=QH({inputs:{x:i},backend:n,attrs:{reductionIndices:o,keepDims:!1}}),c=il(s.shape,o),l=Z({inputs:{x:s},backend:n,attrs:{shape:c}}),u=EU({inputs:{a:i,b:l},backend:n}),d=YV({inputs:{x:u},backend:n}),f=YR({inputs:{x:d},backend:n,attrs:{axis:o,keepDims:!1}}),p=Z({inputs:{x:f},backend:n,attrs:{shape:c}}),m=CU({inputs:{a:d,b:p},backend:n});return n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(u),n.disposeIntermediateTensorInfo(d),n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(p),m}var kU={kernelName:rr,backendName:`webgl`,kernelFunc:OU};function AU(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r,c=s?i:OU({inputs:{logits:i},backend:n,attrs:{dim:i.shape.length-1}}),l=c.shape[0],u=c.shape[1],d=new SU(l,u,a),f=[[o]],p=n.runWebGLProgram(d,[c],`int32`,f);return s||n.disposeIntermediateTensorInfo(c),p}var jU={kernelName:gn,backendName:`webgl`,kernelFunc:AU},MU=VL+`
  return -x;
`,NU=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;function PU(e){let{inputs:t,backend:n}=e,{x:r}=t;if(n.shouldExecuteOnCPU([r])){let[e,t]=tL(n.texData.get(r.dataId).values,r.shape,r.dtype);return n.makeTensorInfo(t,r.dtype,e)}let i;return i=A().getBool(`WEBGL_PACK_UNARY_OPERATIONS`)?new eR(r.shape,NU):new BL(r.shape,MU),n.runWebGLProgram(i,[r],r.dtype)}var FU={kernelName:`Neg`,backendName:`webgl`,kernelFunc:PU},IU=Ap;function LU(e){jr(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=r,{selectedIndices:l}=IU(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c);return n.makeTensorInfo([l.length],`int32`,new Int32Array(l))}var RU={kernelName:yn,backendName:`webgl`,kernelFunc:LU},zU=jp;function BU(e){jr(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,padToMaxOutputSize:l}=r,{selectedIndices:u,validOutputs:d}=zU(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c,l);return[n.makeTensorInfo([u.length],`int32`,new Int32Array(u)),n.makeTensorInfo([],`int32`,new Int32Array([d]))]}var VU={kernelName:bn,backendName:`webgl`,kernelFunc:BU},HU=Mp;function UU(e){jr(`tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=r,{selectedIndices:u,selectedScores:d}=HU(n.readSync(i.dataId),n.readSync(a.dataId),o,s,c,l);return[n.makeTensorInfo([u.length],`int32`,new Int32Array(u)),n.makeTensorInfo([d.length],`float32`,new Float32Array(d))]}var WU={kernelName:xn,backendName:`webgl`,kernelFunc:UU},GU=class{constructor(e,t,n,r){this.variableNames=[`indices`],this.outputShape=[e,t],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${r}), float(${n}),
                      float(index == coords.y)));
      }
    `}},KU={kernelName:Cn,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r,l=C(i.shape),u=new GU(l,o,s,c),d=Z({inputs:{x:i},backend:n,attrs:{shape:[l]}}),f=n.runWebGLProgram(u,[d],a);n.disposeIntermediateTensorInfo(d);let p=[...i.shape,o],m=Z({inputs:{x:f},backend:n,attrs:{shape:p}});return n.disposeIntermediateTensorInfo(f),m}};function qU(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`complex64`){let e=fB({inputs:{input:r},backend:n}),t=qU({inputs:{x:e},backend:n}),i=jB({inputs:{input:r},backend:n}),a=qU({inputs:{x:i},backend:n}),o=vR({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return oH({attrs:{shape:r.shape,dtype:r.dtype,value:r.dtype===`string`?``:0},backend:n})}var JU={kernelName:wr,backendName:`webgl`,kernelFunc:qU};function YU(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`onesLike is not supported under string dtype`);if(r.dtype===`complex64`){let e=fB({inputs:{input:r},backend:n}),t=YU({inputs:{x:e},backend:n}),i=jB({inputs:{input:r},backend:n}),a=qU({inputs:{x:i},backend:n}),o=vR({inputs:{real:t,imag:a},backend:n});return n.disposeIntermediateTensorInfo(e),n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(a),o}return oH({attrs:{shape:r.shape,dtype:r.dtype,value:1},backend:n})}var XU={kernelName:Sn,backendName:`webgl`,kernelFunc:YU};function ZU(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return ZV({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{x(a,e.shape,`All tensors passed to stack must have matching shapes`),b(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=FB({inputs:t.map(e=>{let t=ZV({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeIntermediateTensorInfo(e)),c}var QU={kernelName:wn,backendName:`webgl`,kernelFunc:ZU},$U=class{constructor(e,t,n){this.variableNames=[`x`],this.customUniforms=[{name:`value`,type:`float`}],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=GF(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=[`coords[0]`,`coords[1]`,`coords[2]`,`coords[3]`].slice(0,r);if(r===1){this.userCode=`
        int start = ${a};
        int end = ${o};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;return}this.userCode=`
      ${i} start = ${i}(${a});
      ${i} end = ${i}(${o});

      void main() {
        ${i} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${i} coords = outC - start;
          setOutput(getX(${s}));
        }
      }
    `}},eW=class{constructor(e,t,n){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:`value`,type:`float`}],this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]);let r=e.length,i=GF(r),a=t.map(e=>e[0]).join(`,`),o=t.map((t,n)=>t[0]+e[n]).join(`,`),s=OL(`rc`,r),c=OL(`source`,r),l=`${s[r-1]} < ${this.outputShape[r-1]}`,u=r===1?`source`:`vec2(${c.slice(-2).join()})`,d=[`${i} rc = outputLoc;`,`${s[r-1]} += 1;
       if(${l}) {
      `,r===1?``:`}
       rc = outputLoc;
       ${s[r-2]} += 1;
       if(${s[r-2]} < ${this.outputShape[r-2]}) {`,r===1?``:`  ${s[r-1]} += 1;
         if(${l}) {`],f=r===1?`rc < start || rc >= end`:`any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))`,p=``;for(let e=0,t=r===1?2:4;e<t;e++)p+=`
        ${d[e]}
        if (${f}) {
          result[${e}] = float(value);
        } else {
          ${i} source = rc - start;
          result[${e}] = getChannel(getX(${c.join()}), ${u});
        }
      `;p+=r===1?`} `:`}}`,this.userCode=`
      const ${i} start = ${i}(${a});
      const ${i} end = ${i}(${o});

      void main() {
        ${i} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${p}
        setOutput(result);
      }
    `}},tW=e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:a,constantValue:o}=r;if(C(i.shape)===0)return oH({backend:n,attrs:{shape:a.map((e,t)=>e[0]+i.shape[t]+e[1]),value:o,dtype:i.dtype}});let s=A().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new eW(i.shape,a,o):new $U(i.shape,a,o),c=[[o]];return n.runWebGLProgram(s,[i],i.dtype,c)},nW={kernelName:Tn,backendName:`webgl`,kernelFunc:tW},rW={kernelName:`Pow`,backendName:`webgl`,kernelFunc:AR({opSnippet:`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,packedOpSnippet:`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+mR+`
  return result;
`})};function iW(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r,s=i.shape.length,c=[],l=O(a,i.shape),u=l,d=ol(u,s),f=i;d!=null&&(f=ZR({inputs:{x:i},backend:n,attrs:{perm:d}}),u=cl(u.length,s),c.push(f)),al(`prod`,u,s);let p;if(n.shouldExecuteOnCPU([f])){let e=n.texData.get(f.dataId).values,{outVals:t,outShape:r,outDtype:i}=rL(f.shape,f.dtype,e,u);p=n.makeTensorInfo(r,i,t)}else{let[e,t]=rl(f.shape,u),r=C(t),a=Z({inputs:{x:f},backend:n,attrs:{shape:[-1,r]}}),o=UR(a,Hi(i.dtype),`prod`,n);p=Z({inputs:{x:o},backend:n,attrs:{shape:e}}),c.push(a),c.push(o)}if(o){c.push(p);let e=il(p.shape,l);p=Z({inputs:{x:p},backend:n,attrs:{shape:e}})}return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),p}var aW={kernelName:Dn,backendName:`webgl`,kernelFunc:iW};function oW(e){let{inputs:t,backend:n,attrs:r}=e,{paramsNestedSplits:i,paramsDenseValues:a,indices:o}=t,{outputRaggedRank:s}=r,c=i.map(e=>n.readSync(e.dataId)),l=i.map(e=>e.shape),u=n.readSync(a.dataId),d=n.readSync(o.dataId),[f,p,m]=iL(c,l,u,a.shape,a.dtype,d,o.shape,s),h=f.map(e=>n.makeTensorInfo([e.length],`int32`,e)),g=n.makeTensorInfo(m,a.dtype,p);return h.concat([g])}var sW={kernelName:On,backendName:`webgl`,kernelFunc:oW};function cW(e){let{inputs:t,backend:n}=e,{starts:r,limits:i,deltas:a}=t,o=n.readSync(r.dataId),s=n.readSync(i.dataId),c=n.readSync(a.dataId),[l,u]=aL(o,r.shape,r.dtype,s,i.shape,c,a.shape);return[n.makeTensorInfo([l.length],`int32`,l),n.makeTensorInfo([u.length],r.dtype,u)]}var lW={kernelName:kn,backendName:`webgl`,kernelFunc:cW};function uW(e){let{inputs:t,backend:n,attrs:r}=e,{shape:i,values:a,defaultValue:o,rowPartitionTensors:s}=t,{rowPartitionTypes:c}=r,l=n.readSync(i.dataId),u=n.readSync(a.dataId),d=n.readSync(o.dataId),f=s.map(e=>n.readSync(e.dataId)),p=s.map(e=>e.shape),[m,h]=oL(l,i.shape,u,a.shape,a.dtype,d,o.shape,f,p,c);return n.makeTensorInfo(m,a.dtype,h)}var dW={kernelName:An,backendName:`webgl`,kernelFunc:uW},fW=e=>{let{backend:t,attrs:n}=e,{start:r,stop:i,step:a,dtype:o}=n,s=sL(r,i,a,o);return t.makeTensorInfo([s.length],o,s)},pW={kernelName:jn,backendName:`webgl`,kernelFunc:fW},mW={kernelName:Nn,backendName:`webgl`,kernelFunc:kR({opSnippet:`return 1.0 / x;`})},hW={kernelName:Pn,backendName:`webgl`,kernelFunc:kR({opSnippet:VL+`
  return (x < 0.0) ? 0.0 : x;
`,packedOpSnippet:`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`})},gW={kernelName:Bn,backendName:`webgl`,kernelFunc:kR({opSnippet:VL+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,packedOpSnippet:`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`})},_W=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d;d=i?`(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)`:`vec2(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${l[0]/u[0]},
          ${l[1]/u[1]});
      const vec2 inputShapeRC = vec2(${o}.0, ${s}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}},vW=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d;d=i?`(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)`:`vec3(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${l[0]/u[0]},
          ${l[1]/u[1]},
          ${l[1]/u[1]});
      const vec3 inputShapeRC = vec3(${o}.0, ${s}.0,
                                     ${s}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${n-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}};function yW(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,u=A().getBool(`WEBGL_PACK_IMAGE_OPERATIONS`)?new vW(i.shape,c,l,a,o):new _W(i.shape,c,l,a,o);return n.runWebGLProgram(u,[i],`float32`)}var bW={kernelName:Rn,backendName:`webgl`,kernelFunc:yW},xW=class{constructor(e,t,n){this.variableNames=[`dy`],this.outputShape=[],this.outputShape=t;let[,r,i]=t,[,a,o]=e,s=[n&&a>1?r-1:r,n&&o>1?i-1:i],c=[n&&a>1?a-1:a,n&&o>1?o-1:o],l=s[0]/c[0],u=s[1]/c[1],d=1/l,f=1/u,p=Math.ceil(d)*2+2,m=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${l});
        const float widthScale = float(${u});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${o}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${r-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${i-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}};function SW(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=new xW(a.shape,i.shape,o);return n.runWebGLProgram(s,[a],a.dtype)}var CW={kernelName:zn,backendName:`webgl`,kernelFunc:SW},wW=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d=r?`0.5`:`0.0`,f;f=i?`max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))`:`vec2(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${l[0]/u[0]},
          ${l[1]/u[1]});
      const vec2 inputShapeRC = vec2(${o}.0, ${s}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}},TW=class{constructor(e,t,n,r,i){this.variableNames=[`A`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];let[a,o,s,c]=e;this.outputShape=[a,t,n,c];let l=[r&&t>1?o-1:o,r&&n>1?s-1:s],u=[r&&t>1?t-1:t,r&&n>1?n-1:n],d=r?`0.5`:`0.0`,f;f=i?`max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))`:`vec3(yRC) * effectiveInputOverOutputRatioRC`,this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${l[0]/u[0]},
          ${l[1]/u[1]},
          ${l[1]/u[1]});
      const vec3 inputShapeRC = vec3(${o}.0, ${s}.0,
                                     ${s}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${n-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}};function EW(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,u=A().getBool(`WEBGL_PACK_IMAGE_OPERATIONS`)?new TW(i.shape,c,l,a,o):new wW(i.shape,c,l,a,o);return n.runWebGLProgram(u,[i],i.dtype)}var DW={kernelName:In,backendName:`webgl`,kernelFunc:EW},OW=class{constructor(e,t,n){this.variableNames=[`dy`],this.outputShape=[],this.outputShape=t;let[,r,i]=t,[,a,o]=e,s=[n&&a>1?r-1:r,n&&o>1?i-1:i],c=[n&&a>1?a-1:a,n&&o>1?o-1:o],l=s[0]/c[0],u=s[1]/c[1],d=1/l,f=1/u,p=Math.ceil(d)*2+2,m=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${l});
        const float widthScale = float(${u});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${o}) {
              continue;
            }

            float sourceFracRow =
              float(${s[0]}) *
                (float(dyR) / float(${c[0]}));

            float sourceFracCol =
                float(${s[1]}) *
                  (float(dyC) / float(${c[1]}));

            int sourceNearestRow = int(min(
                float(int(${r}) - 1),
                ${n} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${i}) - 1),
                ${n} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}};function kW(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,s=new OW(a.shape,i.shape,o);return n.runWebGLProgram(s,[a],a.dtype)}var AW={kernelName:Ln,backendName:`webgl`,kernelFunc:kW},jW=class{constructor(e,t){this.variableNames=[`x`];let n=e.length;if(n>4)throw Error(`WebGL backend: Reverse of rank-${n} tensor is not yet supported`);if(this.outputShape=e,n===1){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${e[0]} - coord - 1));
        }
      `;return}let r=n=>t.indexOf(n)!==-1&&e[n]!==1?`${e[n]} - coords[${n}] - 1`:`coords[${n}]`,i=e.map((e,t)=>r(t)).join(`,`),a=GF(n);this.userCode=`
      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${i}));
      }
    `}},MW=class{constructor(e,t){this.variableNames=[`x`],this.packedInputs=!0,this.packedOutput=!0;let n=e.length;if(n>4)throw Error(`WebGL backend: Reverse of rank-${n} tensor is not yet supported`);this.outputShape=e;let r=OL(`rc`,n),i=`${r[n-1]} + 1 < ${this.outputShape[n-1]}`,a=`${r[n-2]} + 1 < ${this.outputShape[n-2]}`,o=GF(n);this.userCode=n===1?`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${e[0]} - rc - 1),
            ${e[0]} - rc - 1);
          if(${i}){
              result.g = getChannel(getX(${e[0]} - (rc  + 1) - 1),
                ${e[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:`
        void main() {
          ${o} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${s(r.slice())};
          if(${i}){
            result.g = ${c(r.slice())};
          }
          if(${a}) {
            result.b = ${l(r.slice())};
            if(${i}) {
              result.a = ${u(r.slice())};
            }
          }
          setOutput(result);
        }
    `;function s(e){return d(e)}function c(e){return e[n-1]=`(`+e[n-1]+` + 1)`,d(e)}function l(e){return e[n-2]=`(`+e[n-2]+` + 1)`,d(e)}function u(e){return e[n-1]=`(`+e[n-1]+` + 1)`,e[n-2]=`(`+e[n-2]+` + 1)`,d(e)}function d(t){let n=e.map((e,n)=>f(n,t));return`getChannel(getX(${n.join(`,`)}), vec2(${n.slice(-2).join(`,`)}))`}function f(n,r){return t.indexOf(n)!==-1&&e[n]!==1?`${e[n]} - ${r[n]} - 1`:`${r[n]}`}}};function NW(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r,o=i.shape.length,s=O(a,i.shape);if(o===0)return gR({inputs:{x:i},backend:n});let c=A().getBool(`WEBGL_PACK_ARRAY_OPERATIONS`)?new MW(i.shape,s):new jW(i.shape,s);return n.runWebGLProgram(c,[i],i.dtype)}var PW={kernelName:Vn,backendName:`webgl`,kernelFunc:NW},FW=class{constructor(e,t){this.variableNames=[`Image`],this.outputShape=[],this.customUniforms=[{name:`params`,type:`vec4`}];let n=e[1],r=e[2];this.outputShape=e;let i=``;i=typeof t==`number`?`float outputValue = ${t.toFixed(2)};`:`
        vec3 fill = vec3(${t.join(`,`)});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${i}
          if(coordX >= 0 && coordX < ${r} && coordY >= 0 && coordY < ${n}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}},IW={kernelName:Dr,backendName:`webgl`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,{radians:i,fillValue:a,center:o}=t,s=n,c=new FW(r.shape,a),[l,u]=Eh(o,r.shape[1],r.shape[2]),d=[[l,u,Math.sin(i),Math.cos(i)]];return s.runWebGLProgram(c,[r],r.dtype,d)}},LW={kernelName:Hn,backendName:`webgl`,kernelFunc:kR({opSnippet:`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`})},RW={kernelName:Un,backendName:`webgl`,kernelFunc:kR({opSnippet:`return inversesqrt(x);`,cpuKernelImpl:cL})},zW=class{constructor(e,t,n,r,i,a,o=!0,s=!1){this.variableNames=[`updates`,`indices`,`defaultValue`],this.outputShape=a;let c=GF(i.length),l=GF(a.length),u=``;n===1?u=`i`:n===2&&(u=`i, j`);let d=`getIndices(${u})`,f=``;r===1?f=`i`:r===2&&(f=`i, coords[1]`);let p=`getUpdates(${f})`,m=``;s&&(m=`coords[0], coords[1]`);let h=`getDefaultValue(${m})`,g=t>1?`strides[j]`:`strides`;this.userCode=`
        ${c} strides = ${c}(${i});

        void main() {
          ${l} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${e}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${t}; j++) {
              int index = round(${d});
              flattenedIndex += index * ${g};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${p};
              found = true;
            }
          }
          setOutput(mix(${h}, sum, float(found)));
        }
      `}},BW=class{constructor(e,t,n,r,i,a,o=!0,s=!1){this.variableNames=[`updates`,`indices`,`defaultValue`],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=a;let c=GF(i.length),l=GF(a.length),u=``;n===1?u=`i`:n===2&&(u=`i, j`);let d=`getIndices(${u})`,f=``;r===1?f=`i`:r===2&&(f=`i, coords[1]`);let p=`getUpdates(${f})`,m=``;s&&(m=`coords[0], coords[1]`);let h=`getDefaultValue(${m})`,g=t>1?`strides[j]`:`strides`,_=t>1?`strides[j + 1]`:`strides`;this.userCode=`
        ${c} strides = ${c}(${i});

        void main() {
          ${l} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${e}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${t}; j+=2) {
              ivec4 index = round(${d});
              flattenedIndex += index.xz * ${g};
              if (j + 1 < ${t}) {
                flattenedIndex += index.yw * ${_};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${p};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${h}, sum, found));
        }
      `}};function VW(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i,updates:a}=t,{shape:o}=r,{sliceRank:s,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=Ff(a,i,o),f=[d/l,l];if(d===0)return n.makeTensorInfo(o,i.dtype);let p=Z({inputs:{x:i},backend:n,attrs:{shape:[c,s]}}),m=Z({inputs:{x:a},backend:n,attrs:{shape:[c,l]}}),h=n.makeTensorInfo([],`float32`,new Float32Array([0])),g;g=A().getBool(`WEBGL_PACK`)?new BW(c,s,p.shape.length,m.shape.length,u,f):new zW(c,s,p.shape.length,m.shape.length,u,f);let _=n.runWebGLProgram(g,[m,p,h],m.dtype),v=Z({inputs:{x:_},backend:n,attrs:{shape:o}});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(_),n.disposeIntermediateTensorInfo(h),v}var HW={kernelName:Wn,backendName:`webgl`,kernelFunc:VW},UW=class{constructor(e,t,n,r){this.variableNames=[`sortedSequence`,`values`],this.customUniforms=[{name:`numInputs`,type:`int`}],this.outputShape=[e,n];let i=`for (int i = 0; i < ${Math.ceil(Math.log2(t+1))}; ++i) { if (left >= right) break;`,a=A().getNumber(`WEBGL_VERSION`)===2?`while (left < right) {`:i,o=r===`left`?`<`:`<=`;this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${a}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${o} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}};function WW(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r,s=new UW(i.shape[0],i.shape[1],a.shape[1],o),c=[[i.shape[1]]];return n.runWebGLProgram(s,[i,a],`int32`,c)}var GW={kernelName:Kn,backendName:`webgl`,kernelFunc:WW},KW=class{constructor(e,t,n){this.variableNames=[`c`,`a`,`b`],this.outputShape=t;let r,i;if(n>4)throw Error(`Where for rank ${n} is not yet supported`);if(n===1)i=`resRC`,r=`resRC`;else{let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],a=[],o=[];for(let r=0;r<t.length;r++)o.push(`${n[r]}`),r<e&&a.push(`${n[r]}`);r=a.join(),i=o.join()}let a=GF(n);this.userCode=`
      void main() {
        ${a} resRC = getOutputCoords();
        float cVal = getC(${r});
        if (cVal >= 1.0) {
          setOutput(getA(${i}));
        } else {
          setOutput(getB(${i}));
        }
      }
    `}};function qW(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t,o=new KW(r.shape.length,i.shape,i.shape.length);return n.runWebGLProgram(o,[r,i,a],Vi(i.dtype,a.dtype))}var JW={kernelName:qn,backendName:`webgl`,kernelFunc:qW},YW={kernelName:Jn,backendName:`webgl`,kernelFunc:kR({opSnippet:`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${Mh};
  float scale = ${Nh};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`})},XW={kernelName:Qn,backendName:`webgl`,kernelFunc:kR({opSnippet:OR+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,packedOpSnippet:`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:uL})},ZW={kernelName:Zn,backendName:`webgl`,kernelFunc:kR({opSnippet:`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`})},QW={kernelName:`Sin`,backendName:`webgl`,kernelFunc:kR({opSnippet:OR+`
  return sin(x);
`,packedOpSnippet:`
  vec4 result = sin(x);
  bvec4 isNaN = isnan(x);
  ${mR}
  return result;
`})},$W={kernelName:Xn,backendName:`webgl`,kernelFunc:kR({opSnippet:`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`})},eG={kernelName:$n,backendName:`webgl`,kernelFunc:kR({opSnippet:`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`})},tG={kernelName:tr,backendName:`webgl`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r;b(i.shape.length<=4,()=>`spaceToBatchND for rank > 4 with a WebGL backend not implemented yet`);let s=a.reduce((e,t)=>e*t),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=[],u=tW({inputs:{x:i},backend:n,attrs:{paddings:c,constantValue:0}}),d=Dh(u.shape,a,s,!1),f=Oh(d.length,a.length,!1),p=kh(u.shape,a,s,!1),m=Z({inputs:{x:u},backend:n,attrs:{shape:d}}),h=ZR({inputs:{x:m},backend:n,attrs:{perm:f}}),g=Z({inputs:{x:h},backend:n,attrs:{shape:p}});return l.push(u),l.push(m),l.push(h),l.forEach(e=>n.disposeIntermediateTensorInfo(e)),g}};function nG(e){let{inputs:t,backend:n}=e,{indices:r,values:i,denseShape:a,defaultValue:o}=t;if(a.shape.length!==1)throw Error(`Dense shape must be a vector, saw:
         ${a.shape}`);if(r.shape.length!==2)throw Error(`Indices must be a matrix, saw:
         ${r.shape}`);if(i.shape.length!==1)throw Error(`Values must be a vector, saw:
         ${i.shape}`);if(o.shape.length!==0)throw Error(`Default value must be a scalar, saw:
        ${o.shape}`);let s=n.readSync(r.dataId),c=n.readSync(i.dataId),l=n.readSync(a.dataId),u=n.readSync(o.dataId)[0],[d,f,p,m,h]=pL(s,r.shape,r.dtype,c,i.dtype,l,u);return[n.makeTensorInfo(f,r.dtype,d),n.makeTensorInfo([f[0]],i.dtype,p),n.makeTensorInfo([m.length],`bool`,new Uint8Array(m.map(e=>Number(e)))),n.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}var rG={kernelName:ir,backendName:`webgl`,kernelFunc:nG};function iG(e){let{inputs:t,backend:n}=e,{inputIndices:r,inputShape:i,newShape:a}=t;if(r.shape.length!==2)throw Error(`Input indices should be a matrix but received shape ${r.shape}`);if(i.shape.length!==1)throw Error(`Input shape should be a vector but received shape ${i.shape}`);if(a.shape.length!==1)throw Error(`Target shape should be a vector but received shape ${a.shape}`);let o=Array.from(n.readSync(i.dataId)),s=n.readSync(r.dataId),c=Array.from(n.readSync(a.dataId)),[l,u,d]=mL(s,r.shape,r.dtype,o,c);return[n.makeTensorInfo(u,r.dtype,l),n.makeTensorInfo([d.length],a.dtype,new Int32Array(d))]}var aG={kernelName:ar,backendName:`webgl`,kernelFunc:iG};function oG(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
              ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
              ${a.shape}`);let o=n.readSync(r.dataId),s=n.readSync(i.dataId),c=n.readSync(a.dataId),[l,u]=hL(o,r.shape,r.dtype,s,c,!0);return n.makeTensorInfo(u,r.dtype,l)}var sG={kernelName:or,backendName:`webgl`,kernelFunc:oG};function cG(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;if(r.shape.length<1)throw Error(`Data should be at least 1 dimensional but received scalar`);if(i.shape.length!==1)throw Error(`Indices should be a vector but received shape
             ${i.shape}`);if(a.shape.length!==1)throw Error(`Segment ids should be a vector but received shape
             ${a.shape}`);let o=n.readSync(r.dataId),s=n.readSync(i.dataId),c=n.readSync(a.dataId),[l,u]=hL(o,r.shape,r.dtype,s,c);return n.makeTensorInfo(u,r.dtype,l)}var lG={kernelName:sr,backendName:`webgl`,kernelFunc:cG};function uG(e){let{inputs:t,backend:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:o}=t,{outputShape:s}=r,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=Ff(a,i,s);if(a.dtype===`string`){let e=lL(n.bufferSync(i),n.bufferSync(a),s,f,u,l,c,d,di(n.readSync(o.dataId)[0]),!1);return n.makeTensorInfo(s,e.dtype,e.values)}let p=new zW(l,c,i.shape.length,a.shape.length,d,[f,1],!1),m=n.runWebGLProgram(p,[a,i,o],a.dtype),h=Z({inputs:{x:m},backend:n,attrs:{shape:s}});return n.disposeIntermediateTensorInfo(m),h}var dG={kernelName:cr,backendName:`webgl`,kernelFunc:uG};function fG(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=r,s=O(o,i.shape)[0],c=ig(i,a,s),l=i.shape.length,u=Array(l).fill(0),d=i.shape.slice();return c.map(e=>{let t=[...d];t[s]=e;let r=$z({inputs:{x:i},backend:n,attrs:{begin:u,size:t}});return u[s]+=e,r})}var pG={kernelName:nr,backendName:`webgl`,kernelFunc:fG},mG=`return sqrt(x);`,hG={kernelName:er,backendName:`webgl`,kernelFunc:kR({opSnippet:mG,packedOpSnippet:mG,cpuKernelImpl:gL})},gG={kernelName:ur,backendName:`webgl`,kernelFunc:kR({opSnippet:`return x * x;`})},_G=`return (a - b) * (a - b);`,vG={kernelName:lr,backendName:`webgl`,kernelFunc:AR({opSnippet:_G,packedOpSnippet:_G})};function yG(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t;if(i.dtype!==`string`)throw Error(`Input must be of datatype string`);let a=_L(Sg(n.readSync(i.dataId)),`string`,r);return n.makeTensorInfo(i.shape,`string`,a)}var bG={kernelName:dr,backendName:`webgl`,kernelFunc:yG};function xG({inputs:e,attrs:t,backend:n}){let{x:r}=e,i=VL+`
    return x > 0.0 ? 1.0 : float(${t.alpha});
  `,a=new BL(r.shape,i);return n.runWebGLProgram(a,[r],r.dtype)}var SG={kernelName:Tr,backendName:`webgl`,kernelFunc:xG},CG=class{constructor(e,t,n){this.variableNames=[`x`],this.outputShape=n;let r=n.length,i=GF(n.length),a=GF(n.length),o=``;if(r===1)o=`coords * strides + begin`;else{let e=0;o=n.map((t,r)=>(e++,n.length===1?`coords * strides[${r}] + begin[${r}]`:`coords[${e-1}] * strides[${r}] + begin[${r}]`)).join(`,`)}this.userCode=`
      ${i} begin = ${i}(${e});
      ${i} strides = ${i}(${t});

      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${o}));
      }
    `}};function wG(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,end:o,strides:s,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:f}=r,{finalShapeSparse:p,finalShape:m,isIdentity:h,sliceDim0:g,isSimpleSlice:_,begin:v,end:y,strides:x}=fh(i.shape,a,o,s,c,l,u,d,f),S;if(h)S=Z({inputs:{x:i},backend:n,attrs:{shape:m}});else if(g||_){b(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let e=$m(v,y,x),t=$z({inputs:{x:i},backend:n,attrs:{begin:v,size:e}});S=Z({inputs:{x:t},backend:n,attrs:{shape:m}}),n.disposeIntermediateTensorInfo(t)}else if(n.shouldExecuteOnCPU([i])){let e=n.readSync(i.dataId),t=vL(p,_o(i.shape,i.dtype,e),x,v);S=n.makeTensorInfo(m,i.dtype,t.values)}else{let e=new CG(v,x,p);S=n.runWebGLProgram(e,[i],i.dtype)}let C=Z({inputs:{x:S},backend:n,attrs:{shape:m}});return n.disposeIntermediateTensorInfo(S),C}var TG={kernelName:fr,backendName:`webgl`,kernelFunc:wG};function EG(e){let{inputs:t,backend:n,attrs:r}=e,{separator:i,nGramWidths:a,leftPad:o,rightPad:s,padWidth:c,preserveShortSequences:l}=r,{data:u,dataSplits:d}=t,[f,p]=yL(n.readSync(u.dataId),n.readSync(d.dataId),i,a,o,s,c,l);return[n.makeTensorInfo([f.length],`string`,f),n.makeTensorInfo(d.shape,`int32`,p)]}var DG={kernelName:pr,backendName:`webgl`,kernelFunc:EG};function OG(e){let{inputs:t,backend:n,attrs:r}=e,{skipEmpty:i}=r,{input:a,delimiter:o}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(a.shape.length!==1)throw Error(`Input must be a vector, got shape: ${a.shape}`);if(o.shape.length!==0)throw Error(`Delimiter must be a scalar, got shape: ${o.shape}`);let s=n.readSync(a.dataId),c=n.readSync(o.dataId)[0],[l,u,d]=bL(s,c,i),f=u.length;return[n.makeTensorInfo([f,2],`int32`,l),n.makeTensorInfo([f],`string`,u),n.makeTensorInfo([2],`int32`,new Int32Array(d))]}var kG={kernelName:mr,backendName:`webgl`,kernelFunc:OG};function AG(e){let{inputs:t,backend:n,attrs:r}=e,{numBuckets:i}=r,{input:a}=t;if(a.dtype!==`string`)throw Error(`Input must be of datatype string`);if(i<=0)throw Error(`Number of buckets must be at least 1`);let o=xL(n.readSync(a.dataId),i);return n.makeTensorInfo(a.shape,`int32`,o)}var jG={kernelName:hr,backendName:`webgl`,kernelFunc:AG},MG={kernelName:`Tan`,backendName:`webgl`,kernelFunc:kR({opSnippet:`return tan(x);`})},NG={kernelName:gr,backendName:`webgl`,kernelFunc:kR({opSnippet:`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`})};function PG(e){let{inputs:t,backend:n,attrs:r}=e,{tensor:i,indices:a,updates:o}=t,{}=r,{sliceRank:s,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=Ff(o,a,i.shape),f=[d/l,l];if(d===0)return n.makeTensorInfo(i.shape,a.dtype);let p=Z({inputs:{x:a},backend:n,attrs:{shape:[c,s]}}),m=Z({inputs:{x:o},backend:n,attrs:{shape:[c,l]}}),h=Z({inputs:{x:i},backend:n,attrs:{shape:f}}),g=new zW(c,s,p.shape.length,m.shape.length,u,f,!1,!0),_=n.runWebGLProgram(g,[m,p,h],h.dtype),v=Z({inputs:{x:_},backend:n,attrs:{shape:i.shape}});return n.disposeIntermediateTensorInfo(p),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(_),v}var FG={kernelName:Gn,backendName:`webgl`,kernelFunc:PG},IG=class{constructor(e,t){this.variableNames=[`A`];let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[r]*t[r];this.outputShape=n,this.rank=n.length;let r=GF(this.rank),i=LG(e);this.userCode=`
      void main() {
        ${r} resRC = getOutputCoords();
        setOutput(getA(${i}));
      }
    `}};function LG(e){let t=e.length;if(t>5)throw Error(`Tile for rank ${t} is not yet supported`);if(t===1)return`imod(resRC, ${e[0]})`;let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`,`resRC.u`],r=[];for(let t=0;t<e.length;t++)r.push(`imod(${n[t]}, ${e[t]})`);return r.join()}function RG(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reps:a}=r;if(i.dtype===`string`||i.shape.length>5){let e=n.readSync(i.dataId),t=i.dtype===`string`?e.map(e=>di(e)):e,r=CL(_o(i.shape,i.dtype,t),a);return n.makeTensorInfo(r.shape,r.dtype,r.values)}let o=new IG(i.shape,a);return n.runWebGLProgram(o,[i],i.dtype)}var zG={kernelName:_r,backendName:`webgl`,kernelFunc:RG},BG=class{constructor(e){this.variableNames=[`x`,`indices`],this.customUniforms=[{name:`n`,type:`int`},{name:`firstPass`,type:`int`},{name:`negativeInf`,type:`float`},{name:`dir`,type:`int`},{name:`inc`,type:`int`}],this.outputShape=e,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}},VG=class{constructor(e){this.variableNames=[`x`,`indices`],this.customUniforms=[{name:`n`,type:`int`},{name:`firstPass`,type:`int`},{name:`k`,type:`int`}],this.outputShape=e,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}};function HG(e,t){t!==null&&e.disposeIntermediateTensorInfo(t)}function UG(e){let t=1;for(;t<e;)t*=2;return t}function WG(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{k:a,sorted:o}=r,s=A().getNumber(`TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD`),c=A().getNumber(`TOPK_K_CPU_HANDOFF_THRESHOLD`),l=i.shape,u=l[l.length-1];if(n.shouldExecuteOnCPU([i])||u<s||a>c){let[e,t]=wL(n.readSync(i.dataId),l,i.dtype,a,o);return[n.makeTensorInfo(e.shape,e.dtype,e.values),n.makeTensorInfo(t.shape,t.dtype,t.values)]}if(a===0)return l[l.length-1]=0,[n.makeTensorInfo(l,i.dtype,[]),n.makeTensorInfo(l,`int32`,[])];if(u===1)return[i,oH({attrs:{shape:l,dtype:`int32`,value:0},backend:n})];let d=n.texData.get(i.dataId),f=d!==null&&d.isPacked,p=f?n.unpackTensor(i):i,m=C(l)/u,h=Z({inputs:{x:p},attrs:{shape:[m,u]},backend:n});f&&HG(n,p);let g=UG(a),_=UG(u),v=null,y=()=>v===null?[h,h]:[h,v],b=(e,t,r)=>{let i=y(),a=new BG(r),o=[[u],[+(v===null)],[-1/0],[e],[t]],s=v;v=n.runWebGLProgram(a,i,`int32`,o),HG(n,s)};for(let e=1;e<g;e*=2){let t=e*2;for(let n=e;n>=1;n/=2)b(t,n,[m,_])}for(let e=_;e>g;e/=2){let t=y(),r=new VG([m,e/2]),i=[[u],[+(v===null)],[g]],a=v;v=n.runWebGLProgram(r,t,`int32`,i),HG(n,a);let o=g/2,s=o*2;for(let e=o;e>=1;e/=2)b(s,e,v.shape)}let x=v;v=$z({inputs:{x:v},backend:n,attrs:{begin:0,size:[m,a]}}),HG(n,x);let S=OH({inputs:{x:h,indices:v},backend:n,attrs:{axis:1,batchDims:1}});HG(n,h);let w=l.slice(0,-1);w.push(a),x=v,v=Z({inputs:{x:v},attrs:{shape:w},backend:n}),HG(n,x);let T=S;return S=Z({inputs:{x:S},attrs:{shape:w},backend:n}),HG(n,T),[S,v]}var GG={kernelName:vr,backendName:`webgl`,kernelFunc:WG},KG=class{constructor(e,t,n,r,i,a){this.variableNames=[`Image`,`Transforms`],this.outputShape=a;let o=n===`nearest`?1:2,s;switch(r){case`constant`:s=1;break;case`reflect`:s=2;break;case`wrap`:s=3;break;case`nearest`:s=4;break;default:s=1}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${s} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${s} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${s} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${e} && 0 <= coordX && coordX < ${t}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${i});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${i});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${t}));
                float mapY = mapCoord(inY, float(${e}));

                if (${o} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}};function qG(e){let{inputs:t,backend:n,attrs:r}=e,{image:i,transforms:a}=t,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=r,[u,d,f,p]=i.shape,[m,h]=l??[d,f],g=new KG(d,f,o,s,c,[u,m,h,p]);return n.runWebGLProgram(g,[i,a],`float32`)}var JG={kernelName:yr,backendName:`webgl`,kernelFunc:qG};function YG(e){let{inputs:t,attrs:n,backend:r}=e,{axis:i}=n,{x:a}=t;JP(a,`unique`),console.warn(`WARNING: `,`UI might be locked temporarily as data is being downloaded`);let{outputValues:o,outputShape:s,indices:c}=EL(r.readSync(a.dataId),i,a.shape,a.dtype);return[r.makeTensorInfo(s,a.dtype,o),r.makeTensorInfo([c.length],`int32`,c)]}var XG={kernelName:xr,backendName:`webgl`,kernelFunc:YG};function ZG(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i,s=o.shape.length,c=i.shape[a],l=Array(s-1),u=0;for(let e=0;e<s;e++)e!==a&&(l[u++]=o.shape[e]);let d=[],f=Array(s).fill(0),p=o.shape.slice();p[a]=1;let m=Array(c);for(let e=0;e<m.length;e++){f[a]=e;let t=$z({inputs:{x:o},backend:n,attrs:{begin:f,size:p}}),r=Z({inputs:{x:t},backend:n,attrs:{shape:l}});m[e]=r,d.push(t)}return d.forEach(e=>n.disposeIntermediateTensorInfo(e)),m}var QG={kernelName:Sr,backendName:`webgl`,kernelFunc:ZG},$G=class{constructor(e,t){this.variableNames=[`x`,`segmentIds`];let n=e.windowSize,r=e.batchSize,i=e.inSize,a=e.numSegments,o=a*Math.ceil(i/n);this.outputShape=[r,o];let s=Math.floor(n/4)*4,c=n%4,l=`
        sumValue += dot(values, segFilter);
    `,u=``;i%n>0&&(u=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return initializationValue;
        }
      `);let d=``;i%n>0&&(d=`
        if (inIdx < 0 || inIdx >= ${i}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = 0.0;

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${d}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${a})) * float(${n}));
        int currentSeg = int(mod(float(outIdx), float(${a})));

        float sumValue = 0.0;

        for (int i = 0; i < ${s}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${l}
        }

        int inIdx = inOffset + ${s};
        if (${c===1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${l}
        } else if (${c===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${l}
        } else if (${c===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${l}
        }
        setOutput(sumValue);
      }
    `}};function eK(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,segmentIds:a}=t,{numSegments:o}=r,s=i.shape.length,c=[],l=0,u=ol([l],s),d=i;u!=null&&(d=ZR({inputs:{x:i},backend:n,attrs:{perm:u}}),c.push(d),l=cl(1,s)[0]);let f=yg(d.shape,l,o),p=C([d.shape[l]]),m=Z({inputs:{x:d},backend:n,attrs:{shape:[-1,p]}});c.push(m);let h=Hi(i.dtype),g=(e,t,r,i,a)=>{let o=e.shape[0],s=e.shape[1],l=vg(s,a),u=new $G({windowSize:l,inSize:s,batchSize:o,numSegments:a},t),d=n.compileAndRun(u,[e,r],i);if(c.push(d),d.shape[1]===a)return d;let f=fW({backend:n,attrs:{start:0,stop:a,step:1,dtype:`float32`}}),p=RG({inputs:{x:f},backend:n,attrs:{reps:[s/l]}});return c.push(f),c.push(p),g(d,t,p,i,a)},_=Z({inputs:{x:g(m,`unsortedSegmentSum`,a,h,o)},backend:n,attrs:{shape:f}}),v=_;if(u!=null){c.push(_);let e=sl(u);v=ZR({inputs:{x:v},backend:n,attrs:{perm:e}})}return c.forEach(e=>n.disposeIntermediateTensorInfo(e)),v}var tK=[tz,iz,az,oz,cz,fz,mz,gz,Cz,Tz,Ez,Dz,Oz,kz,Az,Pz,Iz,Bz,Hz,Wz,qz,tB,rB,sB,lB,_B,yB,CB,yR,DB,IB,GB,ZB,eV,nV,iV,oV,sV,cV,uV,_V,yV,xV,wV,OV,MV,PV,LV,BV,HV,UV,KV,qV,JV,XV,QV,eH,iH,sH,lH,dH,fH,hH,bH,SH,TH,kH,AH,jH,_R,NH,MB,PH,FH,IH,CR,LH,RH,BH,VH,HH,UH,WH,GH,JH,XH,$H,eU,nU,iU,cU,uU,fU,mU,gU,_U,bU,xU,jU,LR,FU,RU,VU,WU,dB,KU,XU,QU,nW,rW,DR,aW,sW,lW,dW,pW,pB,wU,mW,hW,gW,zR,bW,CW,DW,AW,PW,IW,LW,RW,HW,GW,JW,YW,XW,ZW,QW,$W,eB,kU,eG,tG,rG,aG,sG,lG,dG,pG,hG,gG,vG,bG,SG,TG,DG,kG,jG,DU,XR,MG,NG,FG,zG,GG,JG,QR,XG,QG,{kernelName:Cr,backendName:`webgl`,kernelFunc:eK},JU];for(let e of tK)Rr(e);var nK=[{id:`spark-80k`,name:`Spark 80K`,tagline:`Tiny local transformer for instant experiments.`,dModel:48,nLayers:2,nHeads:4,ffDim:192,context:64,defaultBatch:16,defaultLr:.003,risk:`safe`},{id:`pulse-250k`,name:`Pulse 250K`,tagline:`Still quick, noticeably better pattern capacity.`,dModel:88,nLayers:2,nHeads:4,ffDim:352,context:96,defaultBatch:12,defaultLr:.0025,risk:`safe`},{id:`ember-500k`,name:`Ember 500K`,tagline:`Recommended browser starter model.`,dModel:128,nLayers:2,nHeads:4,ffDim:512,context:128,defaultBatch:10,defaultLr:.002,risk:`safe`},{id:`nova-1m`,name:`Nova 1M`,tagline:`Small GPT-style decoder with real attention depth.`,dModel:160,nLayers:3,nHeads:5,ffDim:640,context:160,defaultBatch:8,defaultLr:.0015,risk:`safe`},{id:`vector-5m`,name:`Vector 5M`,tagline:`Heavier local training; use WebGL/GPU.`,dModel:256,nLayers:6,nHeads:8,ffDim:1024,context:256,defaultBatch:4,defaultLr:.001,risk:`caution`},{id:`apex-20m`,name:`Apex 20M`,tagline:`Big for a browser tab. Smaller batches only.`,dModel:512,nLayers:6,nHeads:8,ffDim:2048,context:384,defaultBatch:1,defaultLr:8e-4,risk:`warning`},{id:`titan-75m`,name:`Titan 75M`,tagline:`Serious hardware needed. Can lock up normal laptops.`,dModel:768,nLayers:10,nHeads:12,ffDim:3072,context:512,defaultBatch:1,defaultLr:6e-4,risk:`danger`},{id:`colossus-180m`,name:`Colossus 180M`,tagline:`Workstation/large VRAM territory.`,dModel:1024,nLayers:14,nHeads:16,ffDim:4096,context:768,defaultBatch:1,defaultLr:45e-5,risk:`extreme`},{id:`leviathan-500m`,name:`Leviathan 500M`,tagline:`Experimental monster config. Expect crashes without huge RAM.`,dModel:2048,nLayers:10,nHeads:16,ffDim:8192,context:1024,defaultBatch:1,defaultLr:3e-4,risk:`extreme`},{id:`singularity-1b`,name:`Singularity 1B`,tagline:`The red button: billion-parameter decoder architecture.`,dModel:4096,nLayers:5,nHeads:32,ffDim:16384,context:2048,defaultBatch:1,defaultLr:2e-4,risk:`meltdown`}].map(e=>({...e,estimatedParams:oK(e,256)})),rK=[`
`,`	`,` `,...`abcdefghijklmnopqrstuvwxyz`,...`ABCDEFGHIJKLMNOPQRSTUVWXYZ`,...`0123456789`,...`.,!?;:'"\`~@#$%^&*()-_=+[]{}<>/\\|`,`“`,`”`,`‘`,`’`,`—`,`–`,`…`,`•`,`🙂`,`🔥`,`⚠`,`️`],iK=class e{constructor(e=``,t=256){let n=new Set(rK);for(let t of Array.from(e))n.add(t);n.add(`�`);for(let r of aK(e,t,n)){if(n.size>=t)break;n.add(r)}this.unk=`�`,this.itos=Array.from(n),this.stoi=Object.fromEntries(this.itos.map((e,t)=>[e,t])),this.vocabSize=this.itos.length,this.maxTokenLength=Math.max(1,...this.itos.map(e=>Array.from(e).length))}encode(e){let t=Array.from(e),n=this.stoi[this.unk],r=[];for(let e=0;e<t.length;){let i=null,a=Math.min(this.maxTokenLength,t.length-e);for(let n=a;n>=1;--n){let r=t.slice(e,e+n).join(``);if(this.stoi[r]!==void 0){i=r;break}}i===null?(r.push(n),e+=1):(r.push(this.stoi[i]),e+=Array.from(i).length)}return r}decode(e){return e.map(e=>this.itos[e]??``).join(``).replaceAll(this.unk,``)}toJSON(){return{itos:this.itos,unk:this.unk,maxTokenLength:this.maxTokenLength}}static fromJSON(t){let n=Object.create(e.prototype);return n.itos=t.itos,n.unk=t.unk??`�`,n.stoi=Object.fromEntries(n.itos.map((e,t)=>[e,t])),n.vocabSize=n.itos.length,n.maxTokenLength=t.maxTokenLength??Math.max(1,...n.itos.map(e=>Array.from(e).length)),n}};function aK(e,t,n){let r=new Map,i=e=>{!e||e.length<2||e.length>36||n.has(e)||r.set(e,(r.get(e)??0)+1)};for(let t of[/ ?[A-Za-z][A-Za-z0-9_'’-]{1,}/g,/ ?\d+(?:[.,:]\d+)*/g,/\n\n/g,/ ?[.!?;,]+/g,/ ?[()\[\]{}<>]+/g])for(let n of e.matchAll(t))i(n[0]);for(let t of e.matchAll(/[A-Za-z][A-Za-z0-9_'’-]{4,}/g)){let e=t[0];for(let t=3;t<=Math.min(8,e.length);t+=1)for(let n=0;n<=e.length-t;n+=1)i(e.slice(n,n+t))}return Array.from(r.entries()).filter(([,e])=>e>1).sort((e,t)=>t[1]*Math.log2(t[0].length+1)-e[1]*Math.log2(e[0].length+1)).slice(0,Math.max(0,t-n.size)).map(([e])=>e)}function oK(e,t=256){let n=e.dModel,r=e.ffDim,i=4*n,a=4*(n*n+n),o=n*r+r+r*n+n,s=i+a+o,c=t*n,l=e.context*n,u=2*n,d=n*t+t;return Math.round(c+l+e.nLayers*s+u+d)}function sK(e,t=128,n=128,r=2,i=1){return e*4*4+4*i*r*(t*n*10+t*t*4)}function cK(e){if(!Number.isFinite(e))return`unknown`;let t=[`B`,`KB`,`MB`,`GB`,`TB`],n=e,r=0;for(;n>=1024&&r<t.length-1;)n/=1024,r+=1;return`${n>=10||r===0?n.toFixed(0):n.toFixed(1)} ${t[r]}`}function lK(e){return e>=1e9?`${(e/1e9).toFixed(2)}B`:e>=1e6?`${(e/1e6).toFixed(e>=1e7?0:1)}M`:e>=1e3?`${(e/1e3).toFixed(0)}K`:`${e}`}function uK(e){return e.reduce((e,t)=>e*t,1)}function dK(e,t,n){return Cd(e,0,t,`float32`,n)}function fK(e){return Ru(e,`float32`)}function pK(e){return zu(e,`float32`)}function mK(e){return F(()=>{let t=e.pow(H(3)).mul(.044715),n=e.add(t).mul(Math.sqrt(2/Math.PI));return e.mul(.5).mul(Ms(n).add(1))})}var hK=class{constructor(e,t){if(e.dModel%e.nHeads!==0)throw Error(`dModel (${e.dModel}) must be divisible by nHeads (${e.nHeads}).`);this.config={...e},this.tokenizer=t,this.vocabSize=t.vocabSize,this.scope=`v0id_${Date.now()}_${Math.random().toString(36).slice(2)}`,this.vars={},this.trainableVars=[],this.logicalNameByVariableName=new Map,this.maskCache=new Map,this.optimizer=null,this.optimizerKey=``,this.initVariables()}initVariables(){let{dModel:e,context:t,nLayers:n,ffDim:r}=this.config,i=.02;this.register(`tokenEmbedding`,dK([this.vocabSize,e],i,11)),this.register(`positionEmbedding`,dK([t,e],i,12));for(let t=0;t<n;t+=1){let n=`block${t}`,i=1/Math.sqrt(e);this.register(`${n}/ln1/gamma`,pK([e])),this.register(`${n}/ln1/beta`,fK([e])),this.register(`${n}/attn/wq`,dK([e,e],i,101+t*20)),this.register(`${n}/attn/bq`,fK([e])),this.register(`${n}/attn/wk`,dK([e,e],i,102+t*20)),this.register(`${n}/attn/bk`,fK([e])),this.register(`${n}/attn/wv`,dK([e,e],i,103+t*20)),this.register(`${n}/attn/bv`,fK([e])),this.register(`${n}/attn/wo`,dK([e,e],i,104+t*20)),this.register(`${n}/attn/bo`,fK([e])),this.register(`${n}/ln2/gamma`,pK([e])),this.register(`${n}/ln2/beta`,fK([e])),this.register(`${n}/ffn/w1`,dK([e,r],Math.sqrt(2/e),201+t*20)),this.register(`${n}/ffn/b1`,fK([r])),this.register(`${n}/ffn/w2`,dK([r,e],Math.sqrt(2/r),202+t*20)),this.register(`${n}/ffn/b2`,fK([e]))}this.register(`finalLn/gamma`,pK([e])),this.register(`finalLn/beta`,fK([e])),this.register(`lmHead/w`,dK([e,this.vocabSize],.02,301)),this.register(`lmHead/b`,fK([this.vocabSize]))}register(e,t){let n=Kf(t,!0,`${this.scope}/${e}`);return this.vars[e]=n,this.trainableVars.push(n),this.logicalNameByVariableName.set(n.name,e),t.dispose(),n}parameterCount(){return Object.values(this.vars).reduce((e,t)=>e+uK(t.shape),0)}getCausalMask(e){if(!this.maskCache.has(e)){let t=new Float32Array(e*e);for(let n=0;n<e;n+=1)for(let r=n+1;r<e;r+=1)t[n*e+r]=-1e9;this.maskCache.set(e,ba(Mf(t,[1,1,e,e],`float32`)))}return this.maskCache.get(e)}layerNorm(e,t,n){return F(()=>{let r=e.mean(-1,!0),i=e.sub(r),a=i.square().mean(-1,!0);return i.mul(Bd(a.add(1e-5))).mul(t).add(n)})}attention(e,t){return F(()=>{let{dModel:n,nHeads:r}=this.config,[i,a]=e.shape,o=n/r,s=`block${t}/attn`,c=e.reshape([-1,n]),l=(e,t)=>c.matMul(this.vars[`${s}/${e}`]).add(this.vars[`${s}/${t}`]).reshape([i,a,r,o]).transpose([0,2,1,3]),u=l(`wq`,`bq`),d=l(`wk`,`bk`),f=l(`wv`,`bv`);return Es(sf(Es(u,d,!1,!0).div(Math.sqrt(o)).add(this.getCausalMask(a)),-1),f).transpose([0,2,1,3]).reshape([i*a,n]).matMul(this.vars[`${s}/wo`]).add(this.vars[`${s}/bo`]).reshape([i,a,n])})}feedForward(e,t){return F(()=>{let{dModel:n,ffDim:r}=this.config,[i,a]=e.shape,o=`block${t}/ffn`;return mK(e.reshape([-1,n]).matMul(this.vars[`${o}/w1`]).add(this.vars[`${o}/b1`])).matMul(this.vars[`${o}/w2`]).add(this.vars[`${o}/b2`]).reshape([i,a,n])})}forward(e){return F(()=>{let{dModel:t,context:n,nLayers:r}=this.config,[i,a]=e.shape;if(a>n)throw Error(`Prompt length ${a} exceeds context window ${n}.`);let o=Rl(this.vars.tokenEmbedding,e),s=Rl(this.vars.positionEmbedding,Ed(0,a,1,`int32`)).expandDims(0),c=o.add(s);for(let e=0;e<r;e+=1){let t=`block${e}`,n=this.layerNorm(c,this.vars[`${t}/ln1/gamma`],this.vars[`${t}/ln1/beta`]);c=c.add(this.attention(n,e));let r=this.layerNorm(c,this.vars[`${t}/ln2/gamma`],this.vars[`${t}/ln2/beta`]);c=c.add(this.feedForward(r,e))}return this.layerNorm(c,this.vars[`finalLn/gamma`],this.vars[`finalLn/beta`]).reshape([i*a,t]).matMul(this.vars[`lmHead/w`]).add(this.vars[`lmHead/b`]).reshape([i,a,this.vocabSize])})}lossForBatch(e,t){return F(()=>{let n=this.forward(e),[r,i,a]=n.shape,o=n.reshape([r*i,a]),s=Zu(t.flatten(),a);return Mm.softmaxCrossEntropy(s,o).mean()})}ensureOptimizer(e){let t=`${e}`;(!this.optimizer||this.optimizerKey!==t)&&(this.optimizer=hh.adam(e,.9,.95,1e-8),this.optimizerKey=t)}async trainStep(e,t,n={}){let r=Number(n.learningRate??this.config.defaultLr??.001),i=Number(n.clipNorm??1),a=Number(n.weightDecay??.01);this.ensureOptimizer(r);let{value:o,grads:s}=lu(()=>F(()=>this.lossForBatch(e,t)),this.trainableVars),c=Object.keys(s),l=null,u=null;if(i>0&&c.length>0){l=F(()=>gl(Po(c.map(e=>s[e].square().sum())).add(1e-12)));let e=F(()=>Vu(H(1),H(i).div(l.add(1e-8))));u={};for(let t of c)u[t]=F(()=>s[t].mul(e));e.dispose(),this.optimizer.applyGradients(u)}else this.optimizer.applyGradients(s);if(a>0){let e=Math.max(0,1-r*a);for(let t of this.trainableVars){let n=this.logicalNameByVariableName.get(t.name)??``;n.includes(`/b`)||n.toLowerCase().includes(`ln`)||n.endsWith(`/beta`)||n.endsWith(`/gamma`)||F(()=>t.assign(t.mul(e)))}}let[d,f]=await Promise.all([o.data().then(e=>e[0]),l?l.data().then(e=>e[0]):Promise.resolve(0)]);return o.dispose(),l&&l.dispose(),Object.values(s).forEach(e=>e.dispose()),u&&Object.values(u).forEach(e=>e.dispose()),{loss:d,gradNorm:f,perplexity:Math.exp(Math.min(20,d))}}async generate(e,t={},n=()=>{}){let r=Number(t.maxNewTokens??240),i=Math.max(.05,Number(t.temperature??.85)),a=Math.max(1,Number(t.topK??40)),o=Math.min(1,Math.max(.01,Number(t.topP??.92))),s=Math.max(1,Number(t.repetitionPenalty??1.08)),c=t.stopSequences??[`
User:`,`
USER:`,`
Human:`],l=this.tokenizer.encode(e),u=``;for(let e=0;e<r&&!(t.signal?.aborted||t.shouldStop?.());e+=1){let e=l.slice(-this.config.context),t=jf(e,[1,e.length],`int32`),r=F(()=>this.forward(t).slice([0,e.length-1,0],[1,1,this.vocabSize]).reshape([this.vocabSize]));t.dispose();let d=Array.from(await r.data());r.dispose(),vK(d,l.slice(-128),s);let f=_K(d,{temperature:i,topK:a,topP:o});l.push(f);let p=this.tokenizer.decode([f]);if(u+=p,n(p,u),c.some(e=>u.includes(e))){u=yK(u,c);break}await _h()}return u}async exportJSON(){let e=[];for(let[t,n]of Object.entries(this.vars))e.push({name:t,shape:n.shape,data:Array.from(await n.data())});return{format:`v0idgpt-local-transformer-v1`,createdAt:new Date().toISOString(),config:this.config,tokenizer:this.tokenizer.toJSON(),weights:e}}async importWeights(e){for(let t of e){let e=this.vars[t.name];if(!e)continue;let n=uK(e.shape);if(n!==t.data.length)throw Error(`Weight ${t.name} has ${t.data.length} values, expected ${n}.`);let r=ma(t.data,t.shape,`float32`);e.assign(r),r.dispose(),await _h()}}dispose(){this.optimizer?.dispose&&this.optimizer.dispose(),Object.values(this.vars).forEach(e=>e.dispose()),this.trainableVars=[],this.vars={},this.maskCache.forEach(e=>e.dispose()),this.maskCache.clear()}};function gK(e,t,n){if(e.length<3)throw Error(`Training text is too short. Add at least a few sentences.`);let r=Math.max(1,Math.min(t,e.length-1)),i=new Int32Array(n*r),a=new Int32Array(n*r),o=Math.max(0,e.length-r-1);for(let t=0;t<n;t+=1){let n=o===0?0:Math.floor(Math.random()*(o+1));for(let o=0;o<r;o+=1)i[t*r+o]=e[n+o],a[t*r+o]=e[n+o+1]}return{xs:jf(i,[n,r],`int32`),ys:jf(a,[n,r],`int32`),sequenceLength:r}}function _K(e,{temperature:t=1,topK:n=40,topP:r=.92}={}){let i=e.map((e,n)=>({index:n,value:e/t})).sort((e,t)=>t.value-e.value).slice(0,Math.min(n,e.length)),a=i[0]?.value??0,o=i.map(e=>({...e,probability:Math.exp(e.value-a)})),s=o.reduce((e,t)=>e+t.probability,0)||1;if(o=o.map(e=>({...e,probability:e.probability/s})),r<1){let e=0,t=[];for(let n of o)if(e+=n.probability,t.push(n),e>=r)break;o=t,s=o.reduce((e,t)=>e+t.probability,0)||1,o=o.map(e=>({...e,probability:e.probability/s}))}let c=Math.random();for(let e of o)if(c-=e.probability,c<=0)return e.index;return o[o.length-1]?.index??0}function vK(e,t,n){if(n<=1)return;let r=new Set(t);for(let t of r)t<0||t>=e.length||(e[t]=e[t]>0?e[t]/n:e[t]*n)}function yK(e,t){let n=e.length;for(let r of t){let t=e.indexOf(r);t!==-1&&(n=Math.min(n,t))}return e.slice(0,n)}var bK=`V0idGPT Reborn is a local transformer chatbot. It does not call an external API. It studies the training text, predicts the next token, and generates new text one token at a time.

User: What are you?
Assistant: I am a locally trained decoder-only transformer. I use embeddings, causal self-attention, feed-forward layers, layer normalization, Adam training, and probabilistic sampling.

User: How should you answer?
Assistant: I should answer from patterns I learned in the text. I should be direct, helpful, technical, and honest about uncertainty.

User: Explain local training.
Assistant: Paste text into the training box, choose a model size, train for steps, then chat. More data and more steps improve the generated responses.

A transformer learns by comparing its predicted next token to the real next token. The loss falls when the model gets better. A larger model can learn richer structure, but it also needs much more memory and time.`,Q={backendReady:!1,selectedPreset:nK[0],tokenizer:new iK(`User:
Assistant:
`),model:null,activeConfigId:null,isTraining:!1,abortTraining:!1,generationAbort:null,chatLog:``,trainedSteps:0,lastLoss:null},xK=document.querySelector(`#app`);xK.innerHTML=`
  <div class="shell">
    <header class="hero">
      <div>
        <p class="eyebrow">No API · Real decoder transformer · Runs locally in the browser</p>
        <h1>V0idGPT Reborn</h1>
        <p class="hero-copy">Train a GPT-style hybrid character/subword LLM from scratch, then chat with the weights you just trained. The chatbot generates tokens autoregressively from your input — no canned response table, no external model endpoint.</p>
      </div>
      <div class="runtime-card">
        <span class="status-dot" id="backendDot"></span>
        <strong id="backendName">Starting TensorFlow.js…</strong>
        <small id="runtimeStats">Preparing local matmul backend</small>
      </div>
    </header>

    <main class="grid-layout">
      <section class="panel models-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">1 · Pick architecture</p>
            <h2>10 trainable model sizes</h2>
          </div>
          <button class="ghost-button" id="recalcButton" title="Refresh memory and parameter estimates">Refresh</button>
        </div>
        <div class="model-grid" id="modelGrid"></div>
        <div class="selected-model" id="selectedDetails"></div>
        <label class="unlock-row">
          <input type="checkbox" id="unlockLarge" />
          <span>Unlock high-RAM initialisation attempts for dangerous models</span>
        </label>
        <button class="primary-button full" id="initModel">Initialize selected transformer</button>
      </section>

      <section class="panel train-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">2 · Train locally</p>
            <h2>Chat/train UI</h2>
          </div>
          <span class="pill" id="trainStatePill">Idle</span>
        </div>

        <label class="field-label" for="trainText">Training text</label>
        <textarea id="trainText" spellcheck="false" placeholder="Paste dialogue, notes, docs, code, or any text. The model learns next-token prediction from this data."></textarea>
        <div class="textarea-actions">
          <button class="ghost-button" id="demoCorpus">Insert tiny sanity corpus</button>
          <button class="ghost-button" id="appendChat">Append chat log</button>
          <button class="ghost-button danger-text" id="clearCorpus">Clear</button>
        </div>

        <div class="controls-grid">
          <label>Steps<input id="stepsInput" type="number" min="1" max="100000" value="120" /></label>
          <label>Batch<input id="batchInput" type="number" min="1" max="128" value="16" /></label>
          <label>Learning rate<input id="lrInput" type="number" step="0.0001" min="0.00001" value="0.003" /></label>
          <label>Grad clip<input id="clipInput" type="number" step="0.1" min="0" value="1" /></label>
          <label>AdamW decay<input id="decayInput" type="number" step="0.001" min="0" value="0.01" /></label>
          <label>Report every<input id="reportInput" type="number" min="1" max="1000" value="5" /></label>
        </div>

        <div class="progress-wrap">
          <div class="progress-top">
            <span id="progressLabel">No training run yet.</span>
            <span id="lossLabel">loss: —</span>
          </div>
          <progress id="trainProgress" value="0" max="100"></progress>
          <small id="trainMetrics">Perplexity, tokens/sec, gradient norm, and tensor memory show here while training.</small>
        </div>

        <div class="button-row">
          <button class="primary-button" id="startTrain">Start training</button>
          <button class="secondary-button" id="stopTrain" disabled>Stop</button>
        </div>

        <div class="button-row thin">
          <button class="ghost-button" id="exportModel" disabled>Export weights JSON</button>
          <label class="ghost-button file-button" for="importModel">Import weights JSON</label>
          <input id="importModel" type="file" accept="application/json,.json" hidden />
        </div>
      </section>

      <section class="panel chat-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">3 · Chat with your model</p>
            <h2>Autoregressive generation</h2>
          </div>
          <button class="ghost-button" id="clearChat">Clear chat</button>
        </div>

        <div class="chat-messages" id="chatMessages">
          <div class="system-message">Initialize and train a model, then ask it something. It will generate tokenizer pieces from the transformer logits.</div>
        </div>

        <div class="chat-input-row">
          <textarea id="chatInput" rows="3" placeholder="Message the local model…"></textarea>
          <button class="primary-button send" id="sendChat">Generate</button>
        </div>

        <div class="sampling-controls">
          <label>Max tokens<input id="maxTokensInput" type="number" min="1" max="2000" value="260" /></label>
          <label>Temperature<input id="tempInput" type="number" min="0.05" max="2" step="0.05" value="0.85" /></label>
          <label>Top-K<input id="topKInput" type="number" min="1" max="256" value="40" /></label>
          <label>Top-P<input id="topPInput" type="number" min="0.01" max="1" step="0.01" value="0.92" /></label>
          <label>Repetition penalty<input id="repInput" type="number" min="1" max="2" step="0.01" value="1.08" /></label>
        </div>
        <label class="unlock-row compact">
          <input type="checkbox" id="autoTrainChat" checked />
          <span>Append chat transcript to training box after each generation</span>
        </label>
      </section>

      <section class="panel info-panel">
        <p class="eyebrow">Quality-safe speedups included</p>
        <ul>
          <li>Vectorized TensorFlow.js matmuls with WebGL acceleration when available.</li>
          <li>True causal multi-head self-attention, learned token/position embeddings, GELU FFN, residuals, and layer norm.</li>
          <li>Adam optimizer, gradient clipping, AdamW-style decay, batched random training windows, and aggressive tensor disposal.</li>
          <li>Top-K + nucleus sampling and repetition penalty for faster, cleaner decoding without hardcoded replies.</li>
        </ul>
      </section>
    </main>
  </div>
`;var SK=e=>document.querySelector(e),$={backendDot:SK(`#backendDot`),backendName:SK(`#backendName`),runtimeStats:SK(`#runtimeStats`),modelGrid:SK(`#modelGrid`),selectedDetails:SK(`#selectedDetails`),unlockLarge:SK(`#unlockLarge`),initModel:SK(`#initModel`),recalcButton:SK(`#recalcButton`),trainText:SK(`#trainText`),demoCorpus:SK(`#demoCorpus`),appendChat:SK(`#appendChat`),clearCorpus:SK(`#clearCorpus`),stepsInput:SK(`#stepsInput`),batchInput:SK(`#batchInput`),lrInput:SK(`#lrInput`),clipInput:SK(`#clipInput`),decayInput:SK(`#decayInput`),reportInput:SK(`#reportInput`),trainStatePill:SK(`#trainStatePill`),trainProgress:SK(`#trainProgress`),progressLabel:SK(`#progressLabel`),lossLabel:SK(`#lossLabel`),trainMetrics:SK(`#trainMetrics`),startTrain:SK(`#startTrain`),stopTrain:SK(`#stopTrain`),exportModel:SK(`#exportModel`),importModel:SK(`#importModel`),chatMessages:SK(`#chatMessages`),chatInput:SK(`#chatInput`),sendChat:SK(`#sendChat`),clearChat:SK(`#clearChat`),maxTokensInput:SK(`#maxTokensInput`),tempInput:SK(`#tempInput`),topKInput:SK(`#topKInput`),topPInput:SK(`#topPInput`),repInput:SK(`#repInput`),autoTrainChat:SK(`#autoTrainChat`)};function CK(e){return{safe:`SAFE`,caution:`GPU ADVISED`,warning:`⚠ BIG`,danger:`🚨 DANGER`,extreme:`☢ HUGE`,meltdown:`💀 1B RED BUTTON`}[e]??`LOCAL`}function wK(e,t=e.estimatedParams){return e.risk===`meltdown`?`BILLION-PARAMETER MODE: likely to crash a normal browser tab. Use only on huge RAM/VRAM hardware.`:e.risk===`extreme`?`EXTREME: workstation-class memory required. Export your work before attempting.`:e.risk===`danger`?`DANGER: this can freeze or kill the tab on consumer laptops.`:e.risk===`warning`?`WARNING: reduce batch size and expect slow training.`:t>4e6?`Caution: use WebGL/GPU and keep batches small.`:`Good quick-start size for this in-browser implementation.`}function TK(){$.modelGrid.innerHTML=``;for(let e of nK){let t=document.createElement(`button`),n=sK(e.estimatedParams,e.context,e.dModel,e.nLayers,e.defaultBatch);t.className=`model-card ${e.id===Q.selectedPreset.id?`active`:``} risk-${e.risk}`,t.innerHTML=`
      <span class="badge">${CK(e.risk)}</span>
      <strong>${e.name}</strong>
      <small>${e.tagline}</small>
      <span class="model-meta">${lK(e.estimatedParams)} params · ctx ${e.context} · ${cK(n)} train est.</span>
    `,t.addEventListener(`click`,()=>{Q.selectedPreset=e,$.batchInput.value=e.defaultBatch,$.lrInput.value=e.defaultLr,TK(),EK()}),$.modelGrid.appendChild(t)}}function EK(){let e=Q.selectedPreset,t=Q.tokenizer?.vocabSize??256,n=oK(e,t),r=Number($.batchInput.value||e.defaultBatch||1),i=sK(n,e.context,e.dModel,e.nLayers,r),a=Q.model&&Q.activeConfigId===e.id,o=navigator.deviceMemory?`${navigator.deviceMemory} GB device memory reported`:`device memory unknown`;$.selectedDetails.innerHTML=`
    <div class="selected-title">
      <strong>${e.name}</strong>
      <span class="badge risk-${e.risk}">${CK(e.risk)}</span>
    </div>
    <div class="spec-grid">
      <span><b>${lK(n)}</b><small>estimated params @ vocab ${t}</small></span>
      <span><b>${e.nLayers}</b><small>decoder blocks</small></span>
      <span><b>${e.nHeads}</b><small>attention heads</small></span>
      <span><b>${e.dModel}</b><small>d_model</small></span>
      <span><b>${e.ffDim}</b><small>FFN width</small></span>
      <span><b>${e.context}</b><small>context tokens</small></span>
      <span><b>${cK(i)}</b><small>rough training memory</small></span>
      <span><b>${o}</b><small>browser hint</small></span>
    </div>
    <p class="warning-copy ${e.risk===`safe`?``:`loud`}">${wK(e,n)}</p>
    ${a?`<p class="ready-copy">This architecture is currently initialized.</p>`:``}
  `}async function DK(){$.backendName.textContent=`Selecting fastest local backend…`,$.backendDot.classList.add(`loading`);try{A().set(`WEBGL_PACK`,!0),A().set(`WEBGL_DELETE_TEXTURE_THRESHOLD`,0)}catch{}try{await xa(`webgl`)}catch(e){console.warn(`WebGL backend unavailable, falling back to CPU:`,e),await xa(`cpu`)}await Sa(),Q.backendReady=!0,$.backendDot.classList.remove(`loading`),$.backendDot.classList.add(`ready`),OK()}function OK(){let e=va(),t=Q.model?lK(Q.model.parameterCount()):`no model`;$.backendName.textContent=`Backend: ${Ca()}`,$.runtimeStats.textContent=`${t} · tensors ${e.numTensors} · ${cK(e.numBytes)} tensor memory · trained steps ${Q.trainedSteps}`,$.exportModel.disabled=!Q.model||Q.isTraining}function kK(e,t=!1){$.trainStatePill.textContent=e,$.trainStatePill.classList.toggle(`busy`,t),$.startTrain.disabled=t,$.stopTrain.disabled=!t,$.initModel.disabled=t}function AK(){return $.trainText.value.trim()}function jK(e,t){return t<=5e6?!1:!$.unlockLarge.checked}function MK(e,t){let n=new Set;for(let r of Array.from(e))if(t.stoi[r]===void 0&&n.add(r),n.size>=12)break;return Array.from(n)}async function NK({quiet:e=!1}={}){Q.backendReady||await DK();let t=Q.selectedPreset,n=new iK(`${$.trainText.value}\n${Q.chatLog}\nUser:\nAssistant:\n`),r=oK(t,n.vocabSize),i=Number($.batchInput.value||t.defaultBatch||1),a=sK(r,t.context,t.dModel,t.nLayers,i);if(jK(t,r))return PK(`Locked ${t.name}: ${lK(r)} params needs roughly ${cK(a)} while training. Tick the high-RAM unlock box if you really want to try it.`,`warn`),!1;if(r>2e7&&!window.confirm(`${t.name} is a real ${lK(r)} parameter allocation attempt. Rough training memory: ${cK(a)} plus browser overhead. It may freeze or crash this tab. Continue?`)||Q.model&&!e&&!window.confirm(`Reinitializing destroys the currently loaded weights. Continue?`))return!1;try{return $.initModel.disabled=!0,PK(`Allocating ${t.name} locally…`,`busy`),await _h(),Q.model?.dispose(),Q.tokenizer=n,Q.model=new hK(t,Q.tokenizer),Q.activeConfigId=t.id,Q.trainedSteps=0,Q.lastLoss=null,$.progressLabel.textContent=`${t.name} initialized.`,$.lossLabel.textContent=`loss: —`,PK(`${t.name} ready: ${lK(Q.model.parameterCount())} real trainable parameters allocated.`,`ok`),LK(`${t.name} initialized with ${lK(Q.model.parameterCount())} trainable parameters and vocab size ${Q.tokenizer.vocabSize}.`),TK(),EK(),OK(),!0}catch(e){return console.error(e),PK(`Initialization failed: ${e.message}`,`error`),Q.model=null,Q.activeConfigId=null,!1}finally{$.initModel.disabled=Q.isTraining}}function PK(e,t=`info`){$.trainMetrics.textContent=e,$.trainMetrics.dataset.type=t}async function FK(){if(Q.isTraining)return;let e=AK();if(e.length<20){PK(`Paste more training text first. A transformer needs data, not hardcoded answers.`,`warn`);return}if(!Q.model&&!await NK({quiet:!0}))return;let t=MK(e,Q.tokenizer);if(t.length&&window.confirm(`Your corpus contains characters outside the current tokenizer: ${t.join(` `)}. Reinitialize so the model can learn them? Current weights will reset.`)&&!await NK())return;let n=Q.tokenizer.encode(e),r=Math.max(1,Number($.stepsInput.value||1)),i=Math.max(1,Number($.batchInput.value||Q.selectedPreset.defaultBatch||1)),a=Math.max(1e-6,Number($.lrInput.value||Q.selectedPreset.defaultLr||.001)),o=Math.max(0,Number($.clipInput.value||0)),s=Math.max(0,Number($.decayInput.value||0)),c=Math.max(1,Number($.reportInput.value||5));Q.isTraining=!0,Q.abortTraining=!1,kK(`Training`,!0),$.trainProgress.value=0,PK(`Training started. Loss should trend down as the model learns your text.`,`busy`);let l=performance.now(),u=0;try{for(let e=1;e<=r&&!Q.abortTraining;e+=1){let{xs:t,ys:d,sequenceLength:f}=gK(n,Q.model.config.context,i),p=performance.now(),m=await Q.model.trainStep(t,d,{learningRate:a,clipNorm:o,weightDecay:s});if(t.dispose(),d.dispose(),u+=i*f,Q.trainedSteps+=1,Q.lastLoss=m.loss,e===1||e%c===0||e===r){let t=Math.max(.001,(performance.now()-l)/1e3),n=Math.max(.001,(performance.now()-p)/1e3);$.trainProgress.value=e/r*100,$.progressLabel.textContent=`step ${e}/${r} · ${Math.round(u/t)} tok/s avg · ${Math.round(i*f/n)} tok/s step`,$.lossLabel.textContent=`loss: ${m.loss.toFixed(4)} · ppl: ${m.perplexity.toFixed(1)}`,PK(`grad norm ${m.gradNorm.toFixed(3)} · backend ${Ca()} · tensors ${va().numTensors} · ${cK(va().numBytes)}`,`busy`),OK()}await _h()}Q.abortTraining?(PK(`Training stopped by user. Current weights are kept.`,`warn`),$.progressLabel.textContent=`Training stopped.`):(PK(`Training complete. Chat now uses the weights you trained.`,`ok`),$.progressLabel.textContent=`Finished ${r} steps.`,LK(`Training finished: ${r} steps, last loss ${Q.lastLoss?.toFixed(4)??`n/a`}.`))}catch(e){console.error(e),PK(`Training failed: ${e.message}`,`error`),LK(`Training error: ${e.message}`)}finally{Q.isTraining=!1,Q.abortTraining=!1,kK(`Idle`,!1),OK()}}function IK(e,t=``){let n=$.chatMessages.querySelector(`.system-message`);n&&$.chatMessages.children.length===1&&n.remove();let r=document.createElement(`div`);r.className=`message ${e}`;let i=document.createElement(`span`);i.className=`role-label`,i.textContent=e===`user`?`You`:e===`assistant`?`V0idGPT`:`System`;let a=document.createElement(`div`);return a.className=`message-body`,a.textContent=t,r.append(i,a),$.chatMessages.appendChild(r),$.chatMessages.scrollTop=$.chatMessages.scrollHeight,a}function LK(e){IK(`system`,e)}async function RK(){let e=$.chatInput.value.trim();if(!e||Q.generationAbort)return;if(!Q.model){if(!await NK({quiet:!0}))return;LK(`Model is initialized but not trained yet. Output will be mostly random until you train on text.`)}IK(`user`,e),$.chatInput.value=``;let t=IK(`assistant`,``),n=`${Q.chatLog}User: ${e}\nAssistant:`,r=new AbortController;Q.generationAbort=r,$.sendChat.textContent=`Generating…`,$.sendChat.disabled=!0;try{let i=await Q.model.generate(n,{maxNewTokens:Number($.maxTokensInput.value||260),temperature:Number($.tempInput.value||.85),topK:Number($.topKInput.value||40),topP:Number($.topPInput.value||.92),repetitionPenalty:Number($.repInput.value||1.08),signal:r.signal},(e,n)=>{t.textContent=n,$.chatMessages.scrollTop=$.chatMessages.scrollHeight});t.textContent=i,Q.chatLog=`${n}${i}\n`,$.autoTrainChat.checked&&($.trainText.value=`${$.trainText.value}${$.trainText.value.trim()?`

`:``}User: ${e}\nAssistant: ${i}`),OK()}catch(e){console.error(e),t.textContent=`Generation failed: ${e.message}`}finally{Q.generationAbort=null,$.sendChat.textContent=`Generate`,$.sendChat.disabled=!1}}async function zK(){if(!Q.model)return;let e=Q.model.parameterCount();if(e>1e7&&!window.confirm(`Exporting ${lK(e)} parameters to JSON can be huge and slow. Continue?`))return;PK(`Serializing weights…`,`busy`);let t=await Q.model.exportJSON(),n=new Blob([JSON.stringify(t)],{type:`application/json`}),r=URL.createObjectURL(n),i=document.createElement(`a`);i.href=r,i.download=`${Q.model.config.id??`v0idgpt`}-${Date.now()}.json`,i.click(),URL.revokeObjectURL(r),PK(`Weights exported as JSON.`,`ok`)}async function BK(e){if(e)try{PK(`Reading model JSON…`,`busy`);let t=JSON.parse(await e.text());if(t.format!==`v0idgpt-local-transformer-v1`)throw Error(`Not a V0idGPT Reborn local transformer export.`);let n=iK.fromJSON(t.tokenizer),r=oK(t.config,n.vocabSize);if(jK(t.config,r)){PK(`Import locked: ${lK(r)} parameters. Tick high-RAM unlock first.`,`warn`);return}Q.model?.dispose(),Q.tokenizer=n,Q.model=new hK(t.config,n),await Q.model.importWeights(t.weights),Q.activeConfigId=t.config.id;let i=nK.find(e=>e.id===t.config.id);i&&(Q.selectedPreset=i),PK(`Imported ${lK(Q.model.parameterCount())} trainable parameters.`,`ok`),LK(`Imported model ${t.config.name??t.config.id}.`),TK(),EK(),OK()}catch(e){console.error(e),PK(`Import failed: ${e.message}`,`error`)}finally{$.importModel.value=``}}$.initModel.addEventListener(`click`,()=>NK()),$.recalcButton.addEventListener(`click`,()=>{TK(),EK(),OK()}),$.unlockLarge.addEventListener(`change`,EK),$.batchInput.addEventListener(`input`,EK),$.demoCorpus.addEventListener(`click`,()=>{$.trainText.value=`${$.trainText.value}${$.trainText.value.trim()?`

`:``}${bK}`,EK()}),$.appendChat.addEventListener(`click`,()=>{if(!Q.chatLog.trim()){PK(`No chat log to append yet.`,`warn`);return}$.trainText.value=`${$.trainText.value}${$.trainText.value.trim()?`

`:``}${Q.chatLog.trim()}`}),$.clearCorpus.addEventListener(`click`,()=>{window.confirm(`Clear the training text box?`)&&($.trainText.value=``)}),$.trainText.addEventListener(`input`,EK),$.startTrain.addEventListener(`click`,FK),$.stopTrain.addEventListener(`click`,()=>{Q.abortTraining=!0}),$.sendChat.addEventListener(`click`,RK),$.chatInput.addEventListener(`keydown`,e=>{e.key===`Enter`&&(e.metaKey||e.ctrlKey)&&RK()}),$.clearChat.addEventListener(`click`,()=>{Q.chatLog=``,$.chatMessages.innerHTML=`<div class="system-message">Chat cleared. The model weights are unchanged.</div>`}),$.exportModel.addEventListener(`click`,zK),$.importModel.addEventListener(`change`,e=>BK(e.target.files?.[0])),TK(),EK(),kK(`Idle`,!1),DK(),window.setInterval(OK,2500);