//  VDLPManifoldMesh.swift
//  Height-field VDLP como LowLevelMesh (RealityKit) com update IN-PLACE.
//
//  Checklist de performance aplicado:
//   • grade 128–256²  (aqui: gridN, default 192)
//   • buffers atualizados in-place (replaceUnsafeBytes) — a malha NUNCA é recriada
//   • índices gerados uma única vez (topologia estática)
//   • LOD: reduza gridN por distância de câmera se o framerate cair
//   • meça o framerate ANTES de adicionar efeitos (ver notas no fim)
//
//  Resultado é HIPÓTESE DE DESIGN, não diagnóstico. Todo drill-down
//  deve reancorar na fala literal do paciente (ASL).

import RealityKit
import Metal
import simd

struct VDLPVertex {
    var position: SIMD3<Float>
    var normal:   SIMD3<Float>
    var color:    SIMD3<Float>   // viridis(height)
}

@MainActor
final class VDLPManifoldMesh {

    let gridN: Int
    private var mesh: LowLevelMesh
    private let vertexCount: Int
    private let indexCount: Int

    // buffer CPU reutilizado a cada frame (evita alocação)
    private var scratch: [VDLPVertex]

    init(gridN: Int = 192) throws {
        precondition((128...256).contains(gridN), "grade fora de 128–256²")
        self.gridN = gridN
        self.vertexCount = gridN * gridN
        self.indexCount  = (gridN - 1) * (gridN - 1) * 6
        self.scratch = Array(repeating: VDLPVertex(position: .zero,
                                                   normal: SIMD3(0,1,0),
                                                   color: .zero),
                             count: vertexCount)

        // --- layout do vértice (posição, normal, cor) ---
        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = vertexCount
        desc.indexCapacity  = indexCount
        desc.vertexAttributes = [
            .init(semantic: .position, format: .float3, offset: 0),
            .init(semantic: .normal,   format: .float3, offset: MemoryLayout<SIMD3<Float>>.stride),
            .init(semantic: .color,    format: .float3, offset: MemoryLayout<SIMD3<Float>>.stride * 2),
        ]
        desc.vertexLayouts = [ .init(bufferIndex: 0,
                                     bufferStride: MemoryLayout<VDLPVertex>.stride) ]
        desc.indexType = .uint32

        self.mesh = try LowLevelMesh(descriptor: desc)

        buildIndicesOnce()          // topologia estática: só uma vez
    }

    /// Índices do grid triangulado — escritos UMA vez.
    private func buildIndicesOnce() {
        mesh.withUnsafeMutableIndices { raw in
            let idx = raw.bindMemory(to: UInt32.self)
            var k = 0
            for r in 0..<(gridN - 1) {
                for c in 0..<(gridN - 1) {
                    let i0 = UInt32(r * gridN + c)
                    let i1 = UInt32(r * gridN + c + 1)
                    let i2 = UInt32((r + 1) * gridN + c)
                    let i3 = UInt32((r + 1) * gridN + c + 1)
                    idx[k+0]=i0; idx[k+1]=i2; idx[k+2]=i1
                    idx[k+3]=i1; idx[k+4]=i2; idx[k+5]=i3
                    k += 6
                }
            }
        }
    }

    /// Atualiza SOMENTE as alturas + normais + cor a partir do height-field Z
    /// (Z.count == gridN*gridN, normalizado 0…1). Chamado por frame de deformação.
    /// Custo: O(N²) leve; para on-device pesado, mova esta etapa p/ MPS/compute.
    func update(heightfield Z: [Float], heightScale: Float = 0.6) {
        precondition(Z.count == vertexCount)
        let n = gridN
        let inv = 1.0 / Float(n - 1)

        // 1) posições + cor
        for r in 0..<n {
            for c in 0..<n {
                let i = r * n + c
                let x = (Float(c) * inv - 0.5)          // plano [-0.5, 0.5]
                let z = (Float(r) * inv - 0.5)
                let h = Z[i] * heightScale
                scratch[i].position = SIMD3(x, h, z)
                scratch[i].color = viridis(Z[i])
            }
        }
        // 2) normais por diferença central (barato, estável)
        for r in 0..<n {
            for c in 0..<n {
                let i = r * n + c
                let hL = scratch[max(c-1,0)     + r*n].position.y
                let hR = scratch[min(c+1,n-1)   + r*n].position.y
                let hD = scratch[c + max(r-1,0)*n].position.y
                let hU = scratch[c + min(r+1,n-1)*n].position.y
                let nrm = normalize(SIMD3(hL - hR, 2.0 * inv, hD - hU))
                scratch[i].normal = nrm
            }
        }
        // 3) copia IN-PLACE p/ o buffer da GPU — malha não é recriada
        mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
            raw.copyMemory(from: UnsafeRawBufferPointer(
                start: scratch, count: vertexCount * MemoryLayout<VDLPVertex>.stride))
        }
        // 4) bounds (necessário p/ culling correto)
        mesh.parts.replaceAll([
            LowLevelMesh.Part(indexCount: indexCount,
                              topology: .triangle,
                              bounds: BoundingBox(min: [-0.5, 0, -0.5],
                                                  max: [ 0.5, heightScale, 0.5]))
        ])
    }

    /// Cria a ModelEntity uma vez; reutilize e apenas chame `update` depois.
    func makeEntity() throws -> ModelEntity {
        let resource = try MeshResource(from: mesh)
        var material = PhysicallyBasedMaterial()
        material.roughness = 0.7
        material.metallic  = 0.05
        // vertex colors entram via .color no material se o shader suportar;
        // aqui usamos base neutra e deixamos a cor do vértice modular no CustomMaterial.
        return ModelEntity(mesh: resource, materials: [material])
    }

    // viridis aproximado (5 stops) para colorir por densidade/altura
    private func viridis(_ t: Float) -> SIMD3<Float> {
        let stops: [SIMD3<Float>] = [
            SIMD3(0.15,0.02,0.25), SIMD3(0.13,0.33,0.55), SIMD3(0.13,0.57,0.55),
            SIMD3(0.36,0.78,0.38), SIMD3(0.99,0.91,0.14)
        ]
        let x = max(0, min(1, t)) * Float(stops.count - 1)
        let i = Int(x); let f = x - Float(i)
        let a = stops[i]; let b = stops[min(i+1, stops.count-1)]
        return a + (b - a) * f
    }
}

// -----------------------------------------------------------------------------
// USO (SwiftUI + RealityView):
//
//   struct ManifoldView: View {
//     @State private var mesh = try! VDLPManifoldMesh(gridN: 192)
//     var body: some View {
//       RealityView { content in
//         mesh.update(heightfield: currentZ)         // Z vindo do KDE/MPS
//         let e = try! mesh.makeEntity()
//         content.add(e)
//       } update: { content in
//         // deformação longitudinal: só atualiza os buffers, NÃO recria a malha
//         mesh.update(heightfield: interpolatedZ)
//       }
//     }
//   }
//
// PERFORMANCE — medir ANTES de efeitos:
//   1) Rode com gridN=128 e confirme 90 fps (Vision Pro) / 60 fps (iOS) estáveis.
//   2) Só então suba p/ 192/256 ou adicione bacias (★) e tubo geodésico.
//   3) Se cair, aplique LOD: gridN dinâmico por distância, ou passe o passo 1–2
//      (posições/normais) para um Metal compute kernel / MPS.
//   4) A convolução gaussiana do KDE (gerar Z) deve rodar em MPS/vDSP, não na CPU.
// -----------------------------------------------------------------------------
