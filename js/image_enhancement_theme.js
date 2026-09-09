// DEPRECATED — fonctionnalite migree dans onyx_theme.js.
//
// Sonnet avait cree ce fichier en mode "fallback" quand onyx_theme.js
// etait casse (tronque a mi-fichier). Maintenant que onyx_theme.js est
// repare et que "OnyxImageEnhancementNode" est dans sa liste ALL_NODES, ce
// fichier n'a plus de raison d'exister.
//
// Le garder no-op evite une double-application du theme (qui produisait
// 70 flammes au lieu de 35 et un double-rendu de la glow border).
//
// Pour supprimer ce fichier completement : aucun autre fichier du pack n'y
// reference, donc rm en toute securite.

export {};
